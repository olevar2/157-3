"""
Live Strategy Performance Monitor
Real-time strategy performance monitoring and adjustment system.

This module provides comprehensive monitoring of live trading strategies
with real-time performance tracking and adaptive optimization.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from enum import Enum
import threading
import time

class StrategyStatus(Enum):
    """Strategy status definitions"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    UNDER_REVIEW = "under_review"
    OPTIMIZING = "optimizing"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class StrategyMetrics:
    """Real-time strategy performance metrics"""
    strategy_id: str
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    average_trade_duration_minutes: float
    largest_win: float
    largest_loss: float
    consecutive_losses: int
    risk_score: float

@dataclass
class PerformanceAlert:
    """Performance alert data"""
    alert_id: str
    strategy_id: str
    timestamp: datetime
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    recommended_action: str

@dataclass
class StrategyState:
    """Current state of a trading strategy"""
    strategy_id: str
    name: str
    status: StrategyStatus
    start_time: datetime
    last_update: datetime
    metrics: StrategyMetrics
    active_positions: List[Dict]
    recent_trades: List[Dict]
    alerts: List[PerformanceAlert] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

class LiveStrategyMonitor:
    """
    Real-time strategy performance monitoring system.
    
    Features:
    - Continuous performance tracking
    - Real-time alert generation
    - Adaptive threshold management
    - Performance degradation detection
    - Automated strategy pausing/stopping
    - Live vs backtest comparison
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Monitoring settings
        self.update_interval_seconds = config.get('update_interval_seconds', 30)
        self.alert_cooldown_minutes = config.get('alert_cooldown_minutes', 5)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)  # 5%
        
        # Strategy tracking
        self.monitored_strategies: Dict[str, StrategyState] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_win_rate': config.get('min_win_rate', 0.4),
            'max_drawdown': config.get('max_drawdown', 0.1),
            'min_profit_factor': config.get('min_profit_factor', 1.2),
            'max_risk_score': config.get('max_risk_score', 0.8)
        }
        
        # Callbacks for alerts and actions
        self.alert_callbacks: List[Callable] = []
        self.action_callbacks: Dict[str, Callable] = {}
    
    def add_strategy(self, strategy_id: str, strategy_name: str, parameters: Dict[str, Any] = None) -> None:
        """Add a strategy to monitoring"""
        
        strategy_state = StrategyState(
            strategy_id=strategy_id,
            name=strategy_name,
            status=StrategyStatus.ACTIVE,
            start_time=datetime.now(),
            last_update=datetime.now(),
            metrics=self._initialize_metrics(strategy_id),
            active_positions=[],
            recent_trades=[],
            parameters=parameters or {}
        )
        
        self.monitored_strategies[strategy_id] = strategy_state
        self.logger.info(f"Added strategy {strategy_name} ({strategy_id}) to monitoring")
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from monitoring"""
        if strategy_id in self.monitored_strategies:
            del self.monitored_strategies[strategy_id]
            self.logger.info(f"Removed strategy {strategy_id} from monitoring")
    
    def start_monitoring(self) -> None:
        """Start the real-time monitoring system"""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Live strategy monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Live strategy monitoring stopped")
    
    def update_strategy_trade(self, strategy_id: str, trade_data: Dict[str, Any]) -> None:
        """Update strategy with new trade data"""
        if strategy_id not in self.monitored_strategies:
            self.logger.warning(f"Strategy {strategy_id} not found in monitoring")
            return
        
        strategy = self.monitored_strategies[strategy_id]
        
        # Add to recent trades
        strategy.recent_trades.append(trade_data)
        
        # Keep only last 100 trades for performance
        if len(strategy.recent_trades) > 100:
            strategy.recent_trades = strategy.recent_trades[-100:]
        
        # Update metrics
        self._update_strategy_metrics(strategy)
        
        # Check for alerts
        self._check_performance_alerts(strategy)
        
        strategy.last_update = datetime.now()
    
    def update_strategy_position(self, strategy_id: str, position_data: Dict[str, Any]) -> None:
        """Update strategy with position changes"""
        if strategy_id not in self.monitored_strategies:
            return
        
        strategy = self.monitored_strategies[strategy_id]
        
        # Update active positions
        position_id = position_data.get('position_id')
        if position_data.get('status') == 'closed':
            # Remove closed position
            strategy.active_positions = [
                pos for pos in strategy.active_positions 
                if pos.get('position_id') != position_id
            ]
        else:
            # Update or add position
            existing_pos = next(
                (pos for pos in strategy.active_positions 
                 if pos.get('position_id') == position_id), None
            )
            
            if existing_pos:
                existing_pos.update(position_data)
            else:
                strategy.active_positions.append(position_data)
        
        strategy.last_update = datetime.now()
    
    def get_strategy_status(self, strategy_id: str) -> Optional[StrategyState]:
        """Get current status of a strategy"""
        return self.monitored_strategies.get(strategy_id)
    
    def get_all_strategies_status(self) -> Dict[str, StrategyState]:
        """Get status of all monitored strategies"""
        return self.monitored_strategies.copy()
    
    def pause_strategy(self, strategy_id: str, reason: str = "Manual pause") -> bool:
        """Pause a strategy"""
        if strategy_id not in self.monitored_strategies:
            return False
        
        strategy = self.monitored_strategies[strategy_id]
        strategy.status = StrategyStatus.PAUSED
        
        # Create alert
        alert = PerformanceAlert(
            alert_id=f"pause_{strategy_id}_{int(time.time())}",
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message=f"Strategy paused: {reason}",
            metric_name="status",
            current_value=0,
            threshold_value=0,
            recommended_action="Review strategy performance"
        )
        
        self._send_alert(alert)
        return True
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a paused strategy"""
        if strategy_id not in self.monitored_strategies:
            return False
        
        strategy = self.monitored_strategies[strategy_id]
        if strategy.status == StrategyStatus.PAUSED:
            strategy.status = StrategyStatus.ACTIVE
            self.logger.info(f"Strategy {strategy_id} resumed")
            return True
        
        return False
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def add_action_callback(self, action_type: str, callback: Callable) -> None:
        """Add callback for automated actions"""
        self.action_callbacks[action_type] = callback
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update all strategies
                for strategy_id, strategy in self.monitored_strategies.items():
                    if strategy.status == StrategyStatus.ACTIVE:
                        self._update_strategy_metrics(strategy)
                        self._check_performance_alerts(strategy)
                        self._check_automated_actions(strategy)
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                time.sleep(self.update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval_seconds)
    
    def _initialize_metrics(self, strategy_id: str) -> StrategyMetrics:
        """Initialize metrics for a new strategy"""
        return StrategyMetrics(
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            average_trade_duration_minutes=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            consecutive_losses=0,
            risk_score=0.0
        )
    
    def _update_strategy_metrics(self, strategy: StrategyState) -> None:
        """Update strategy performance metrics"""
        trades = strategy.recent_trades
        
        if not trades:
            return
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        # Calculate P&L metrics
        pnls = [t.get('pnl', 0) for t in trades]
        total_pnl = sum(pnls)
        
        # Calculate daily P&L
        today = datetime.now().date()
        daily_trades = [t for t in trades if t.get('exit_time', datetime.now()).date() == today]
        daily_pnl = sum(t.get('pnl', 0) for t in daily_trades)
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
        
        # Calculate other metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        winning_pnls = [pnl for pnl in pnls if pnl > 0]
        losing_pnls = [abs(pnl) for pnl in pnls if pnl < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Calculate consecutive losses
        consecutive_losses = 0
        for trade in reversed(trades):
            if trade.get('pnl', 0) <= 0:
                consecutive_losses += 1
            else:
                break
        
        # Calculate risk score (simplified)
        risk_score = min(1.0, (current_drawdown / 0.1) + (consecutive_losses / 10))
        
        # Update metrics
        strategy.metrics = StrategyMetrics(
            strategy_id=strategy.strategy_id,
            timestamp=datetime.now(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=self._calculate_sharpe_ratio(pnls),
            profit_factor=profit_factor,
            average_trade_duration_minutes=self._calculate_avg_duration(trades),
            largest_win=max(pnls) if pnls else 0.0,
            largest_loss=min(pnls) if pnls else 0.0,
            consecutive_losses=consecutive_losses,
            risk_score=risk_score
        )
    
    def _check_performance_alerts(self, strategy: StrategyState) -> None:
        """Check for performance-based alerts"""
        metrics = strategy.metrics
        
        # Check win rate
        if metrics.total_trades >= 10 and metrics.win_rate < self.performance_thresholds['min_win_rate']:
            self._create_alert(
                strategy.strategy_id,
                AlertLevel.WARNING,
                f"Low win rate: {metrics.win_rate:.1%}",
                "win_rate",
                metrics.win_rate,
                self.performance_thresholds['min_win_rate'],
                "Consider reviewing strategy parameters"
            )
        
        # Check drawdown
        if metrics.current_drawdown > self.performance_thresholds['max_drawdown']:
            self._create_alert(
                strategy.strategy_id,
                AlertLevel.CRITICAL,
                f"High drawdown: {metrics.current_drawdown:.1%}",
                "drawdown",
                metrics.current_drawdown,
                self.performance_thresholds['max_drawdown'],
                "Consider pausing strategy"
            )
        
        # Check consecutive losses
        if metrics.consecutive_losses >= self.max_consecutive_losses:
            self._create_alert(
                strategy.strategy_id,
                AlertLevel.CRITICAL,
                f"Consecutive losses: {metrics.consecutive_losses}",
                "consecutive_losses",
                metrics.consecutive_losses,
                self.max_consecutive_losses,
                "Strategy may need immediate review"
            )
        
        # Check daily drawdown
        if abs(metrics.daily_pnl) > self.max_daily_drawdown * 10000:  # Assuming $10k account
            self._create_alert(
                strategy.strategy_id,
                AlertLevel.EMERGENCY,
                f"Daily loss limit approached: ${metrics.daily_pnl:.2f}",
                "daily_pnl",
                metrics.daily_pnl,
                -self.max_daily_drawdown * 10000,
                "Consider stopping strategy for the day"
            )
    
    def _check_automated_actions(self, strategy: StrategyState) -> None:
        """Check if automated actions should be triggered"""
        metrics = strategy.metrics
        
        # Auto-pause on excessive drawdown
        if metrics.current_drawdown > 0.15:  # 15% drawdown
            self.pause_strategy(strategy.strategy_id, "Excessive drawdown")
        
        # Auto-pause on daily loss limit
        if abs(metrics.daily_pnl) > self.max_daily_drawdown * 10000:
            self.pause_strategy(strategy.strategy_id, "Daily loss limit reached")
    
    def _create_alert(
        self,
        strategy_id: str,
        level: AlertLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        recommended_action: str
    ) -> None:
        """Create and send a performance alert"""
        
        alert = PerformanceAlert(
            alert_id=f"{strategy_id}_{metric_name}_{int(time.time())}",
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            recommended_action=recommended_action
        )
        
        self._send_alert(alert)
    
    def _send_alert(self, alert: PerformanceAlert) -> None:
        """Send alert to all registered callbacks"""
        
        # Add to strategy alerts
        if alert.strategy_id in self.monitored_strategies:
            strategy = self.monitored_strategies[alert.strategy_id]
            strategy.alerts.append(alert)
            
            # Keep only recent alerts
            if len(strategy.alerts) > 50:
                strategy.alerts = strategy.alerts[-50:]
        
        # Add to global alert history
        self.alert_history.append(alert)
        
        # Send to callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.warning(f"Alert: {alert.message} for strategy {alert.strategy_id}")
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _calculate_avg_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in minutes"""
        if not trades:
            return 0.0
        
        durations = []
        for trade in trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            
            if entry_time and exit_time:
                duration = (exit_time - entry_time).total_seconds() / 60
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts from history"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
