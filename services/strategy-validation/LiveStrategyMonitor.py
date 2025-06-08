"""
Live Strategy Monitor

Real-time monitoring system for active trading strategies with performance tracking,
risk management, alerts, and automated strategy controls.

Features:
- Real-time P&L tracking and risk monitoring
- Automated drawdown and exposure controls
- Performance alerts and notifications
- Strategy health diagnostics
- Live metrics dashboard integration
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import json
import websockets
from concurrent.futures import ThreadPoolExecutor


class StrategyStatus(Enum):
    """Strategy monitoring status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    RISK_LIMIT = "risk_limit"
    

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
    pnl: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_drawdown: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    exposure: float
    risk_score: float
    alerts: List[str] = field(default_factory=list)


@dataclass
class RiskLimits:
    """Risk management limits for strategy monitoring"""
    max_drawdown: float = 0.15  # 15% max drawdown
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_exposure: float = 1.0  # 100% exposure limit
    max_positions: int = 20  # Maximum open positions
    min_win_rate: float = 0.30  # Minimum acceptable win rate
    max_consecutive_losses: int = 5  # Max consecutive losing trades


class LiveStrategyMonitor:
    """
    Advanced real-time strategy monitoring system
    
    Provides comprehensive monitoring of live trading strategies with:
    - Real-time performance tracking
    - Risk management and automated controls
    - Alert system with multiple notification channels
    - Strategy health diagnostics
    - Historical performance analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LiveStrategyMonitor
        
        Args:
            config: Configuration dictionary with monitoring settings
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Strategy monitoring data
        self.monitored_strategies: Dict[str, Dict] = {}
        self.strategy_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.strategy_status: Dict[str, StrategyStatus] = {}
        self.risk_limits: Dict[str, RiskLimits] = {}
        
        # Real-time data streams
        self.price_feeds: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.trade_feeds: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Monitoring controls
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.update_interval = self.config.get('update_interval', 1.0)  # seconds
        
        # Alert system
        self.alert_callbacks: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Performance analytics
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("LiveStrategyMonitor initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('LiveStrategyMonitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_strategy(self, strategy_id: str, config: Dict = None) -> bool:
        """
        Add strategy to monitoring
        
        Args:
            strategy_id: Unique strategy identifier
            config: Strategy-specific configuration
            
        Returns:
            bool: True if successfully added
        """
        try:
            if strategy_id in self.monitored_strategies:
                self.logger.warning(f"Strategy {strategy_id} already being monitored")
                return False
            
            strategy_config = config or {}
            self.monitored_strategies[strategy_id] = {
                'config': strategy_config,
                'start_time': datetime.now(),
                'last_update': None,
                'total_trades': 0,
                'starting_capital': strategy_config.get('starting_capital', 100000),
                'current_capital': strategy_config.get('starting_capital', 100000)
            }
            
            # Set risk limits
            risk_config = strategy_config.get('risk_limits', {})
            self.risk_limits[strategy_id] = RiskLimits(**risk_config)
            
            # Initialize strategy status
            self.strategy_status[strategy_id] = StrategyStatus.ACTIVE
            
            self.logger.info(f"Added strategy {strategy_id} to monitoring")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy_id}: {e}")
            return False
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove strategy from monitoring
        
        Args:
            strategy_id: Strategy identifier to remove
            
        Returns:
            bool: True if successfully removed
        """
        try:
            if strategy_id not in self.monitored_strategies:
                self.logger.warning(f"Strategy {strategy_id} not found in monitoring")
                return False
            
            # Clean up strategy data
            del self.monitored_strategies[strategy_id]
            del self.strategy_metrics[strategy_id]
            del self.strategy_status[strategy_id]
            del self.risk_limits[strategy_id]
            
            self.logger.info(f"Removed strategy {strategy_id} from monitoring")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing strategy {strategy_id}: {e}")
            return False
    
    def update_trade(self, strategy_id: str, trade_data: Dict) -> None:
        """
        Update strategy with new trade data
        
        Args:
            strategy_id: Strategy identifier
            trade_data: Trade information (entry/exit, P&L, etc.)
        """
        try:
            if strategy_id not in self.monitored_strategies:
                self.logger.warning(f"Strategy {strategy_id} not being monitored")
                return
            
            # Store trade data
            trade_data['timestamp'] = datetime.now()
            self.trade_feeds[strategy_id].append(trade_data)
            
            # Update strategy metrics
            self._update_strategy_metrics(strategy_id)
            
            # Check risk limits
            self._check_risk_limits(strategy_id)
            
        except Exception as e:
            self.logger.error(f"Error updating trade for {strategy_id}: {e}")
    
    def update_price(self, symbol: str, price_data: Dict) -> None:
        """
        Update real-time price data
        
        Args:
            symbol: Trading symbol
            price_data: Price information (bid, ask, timestamp)
        """
        try:
            price_data['timestamp'] = datetime.now()
            self.price_feeds[symbol].append(price_data)
            
            # Update unrealized P&L for strategies trading this symbol
            self._update_unrealized_pnl(symbol)
            
        except Exception as e:
            self.logger.error(f"Error updating price for {symbol}: {e}")
    
    def _update_strategy_metrics(self, strategy_id: str) -> None:
        """Update comprehensive strategy metrics"""
        try:
            strategy_data = self.monitored_strategies[strategy_id]
            trades = list(self.trade_feeds[strategy_id])
            
            if not trades:
                return
            
            # Calculate performance metrics
            realized_pnl = sum(trade.get('pnl', 0) for trade in trades if trade.get('closed', False))
            unrealized_pnl = sum(trade.get('unrealized_pnl', 0) for trade in trades if not trade.get('closed', True))
            total_pnl = realized_pnl + unrealized_pnl
            
            # Trade statistics
            closed_trades = [t for t in trades if t.get('closed', False)]
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Drawdown calculation
            equity_curve = self._calculate_equity_curve(strategy_id)
            current_drawdown, max_drawdown = self._calculate_drawdown(equity_curve)
            
            # Risk metrics
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Position exposure
            open_positions = len([t for t in trades if not t.get('closed', True)])
            exposure = sum(abs(t.get('size', 0)) for t in trades if not t.get('closed', True))
            
            # Risk score calculation
            risk_score = self._calculate_risk_score(strategy_id, {
                'current_drawdown': current_drawdown,
                'win_rate': win_rate,
                'exposure': exposure,
                'consecutive_losses': self._count_consecutive_losses(strategy_id)
            })
            
            # Create metrics object
            metrics = StrategyMetrics(
                strategy_id=strategy_id,
                timestamp=datetime.now(),
                pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                open_positions=open_positions,
                total_trades=len(closed_trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                sharpe_ratio=sharpe_ratio,
                exposure=exposure,
                risk_score=risk_score
            )
            
            # Store metrics
            self.strategy_metrics[strategy_id].append(metrics)
            strategy_data['last_update'] = datetime.now()
            strategy_data['current_capital'] = strategy_data['starting_capital'] + total_pnl
            
        except Exception as e:
            self.logger.error(f"Error updating metrics for {strategy_id}: {e}")
    
    def _calculate_equity_curve(self, strategy_id: str) -> List[float]:
        """Calculate equity curve from trade history"""
        trades = list(self.trade_feeds[strategy_id])
        starting_capital = self.monitored_strategies[strategy_id]['starting_capital']
        
        equity_curve = [starting_capital]
        current_equity = starting_capital
        
        for trade in trades:
            if trade.get('closed', False):
                current_equity += trade.get('pnl', 0)
                equity_curve.append(current_equity)
        
        return equity_curve
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> tuple:
        """Calculate current and maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        
        current_drawdown = drawdown.iloc[-1]
        max_drawdown = drawdown.min()
        
        return abs(current_drawdown), abs(max_drawdown)
    
    def _count_consecutive_losses(self, strategy_id: str) -> int:
        """Count consecutive losing trades"""
        trades = list(self.trade_feeds[strategy_id])
        closed_trades = [t for t in trades if t.get('closed', False)]
        
        if not closed_trades:
            return 0
        
        consecutive_losses = 0
        for trade in reversed(closed_trades):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        return consecutive_losses
    
    def _calculate_risk_score(self, strategy_id: str, metrics: Dict) -> float:
        """Calculate comprehensive risk score (0-100, higher = riskier)"""
        risk_limits = self.risk_limits[strategy_id]
        score = 0
        
        # Drawdown risk (0-30 points)
        drawdown_ratio = metrics['current_drawdown'] / risk_limits.max_drawdown
        score += min(30, drawdown_ratio * 30)
        
        # Win rate risk (0-20 points)
        if metrics['win_rate'] < risk_limits.min_win_rate:
            score += 20 * (1 - metrics['win_rate'] / risk_limits.min_win_rate)
        
        # Exposure risk (0-25 points)
        exposure_ratio = metrics['exposure'] / risk_limits.max_exposure
        score += min(25, exposure_ratio * 25)
        
        # Consecutive losses risk (0-25 points)
        loss_ratio = metrics['consecutive_losses'] / risk_limits.max_consecutive_losses
        score += min(25, loss_ratio * 25)
        
        return min(100, score)
    
    def _check_risk_limits(self, strategy_id: str) -> None:
        """Check if strategy exceeds risk limits"""
        if not self.strategy_metrics[strategy_id]:
            return
        
        metrics = self.strategy_metrics[strategy_id][-1]
        risk_limits = self.risk_limits[strategy_id]
        alerts = []
        
        # Check drawdown limits
        if metrics.current_drawdown > risk_limits.max_drawdown:
            alerts.append(f"CRITICAL: Drawdown {metrics.current_drawdown:.2%} exceeds limit {risk_limits.max_drawdown:.2%}")
            self._pause_strategy(strategy_id, "Drawdown limit exceeded")
        
        # Check daily loss limit
        daily_pnl = self._calculate_daily_pnl(strategy_id)
        if daily_pnl < -risk_limits.max_daily_loss:
            alerts.append(f"CRITICAL: Daily loss {daily_pnl:.2%} exceeds limit {risk_limits.max_daily_loss:.2%}")
            self._pause_strategy(strategy_id, "Daily loss limit exceeded")
        
        # Check position limits
        if metrics.open_positions > risk_limits.max_positions:
            alerts.append(f"WARNING: Open positions {metrics.open_positions} exceeds limit {risk_limits.max_positions}")
        
        # Check consecutive losses
        consecutive_losses = self._count_consecutive_losses(strategy_id)
        if consecutive_losses >= risk_limits.max_consecutive_losses:
            alerts.append(f"WARNING: {consecutive_losses} consecutive losses reached")
        
        # Send alerts
        for alert in alerts:
            self._send_alert(strategy_id, alert, AlertLevel.CRITICAL if "CRITICAL" in alert else AlertLevel.WARNING)
    
    def _calculate_daily_pnl(self, strategy_id: str) -> float:
        """Calculate today's P&L as percentage of capital"""
        today = datetime.now().date()
        trades = [t for t in self.trade_feeds[strategy_id] 
                 if t.get('timestamp', datetime.now()).date() == today]
        
        daily_pnl = sum(t.get('pnl', 0) for t in trades if t.get('closed', False))
        starting_capital = self.monitored_strategies[strategy_id]['starting_capital']
        
        return daily_pnl / starting_capital if starting_capital > 0 else 0
    
    def _pause_strategy(self, strategy_id: str, reason: str) -> None:
        """Pause strategy due to risk limit breach"""
        self.strategy_status[strategy_id] = StrategyStatus.RISK_LIMIT
        self.logger.critical(f"Strategy {strategy_id} paused: {reason}")
        
        # Send emergency alert
        self._send_alert(strategy_id, f"STRATEGY PAUSED: {reason}", AlertLevel.EMERGENCY)
    
    def _send_alert(self, strategy_id: str, message: str, level: AlertLevel) -> None:
        """Send alert through configured channels"""
        alert = {
            'timestamp': datetime.now(),
            'strategy_id': strategy_id,
            'level': level.value,
            'message': message
        }
        
        self.alert_history.append(alert)
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.log(
            logging.CRITICAL if level == AlertLevel.EMERGENCY else
            logging.ERROR if level == AlertLevel.CRITICAL else
            logging.WARNING if level == AlertLevel.WARNING else
            logging.INFO,
            f"[{level.value.upper()}] {strategy_id}: {message}"
        )
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get latest metrics for strategy"""
        if strategy_id not in self.strategy_metrics or not self.strategy_metrics[strategy_id]:
            return None
        return self.strategy_metrics[strategy_id][-1]
    
    def get_strategy_status(self, strategy_id: str) -> Optional[StrategyStatus]:
        """Get current strategy status"""
        return self.strategy_status.get(strategy_id)
    
    def get_all_strategies(self) -> Dict[str, Dict]:
        """Get overview of all monitored strategies"""
        overview = {}
        for strategy_id in self.monitored_strategies:
            latest_metrics = self.get_strategy_metrics(strategy_id)
            overview[strategy_id] = {
                'status': self.strategy_status[strategy_id].value,
                'metrics': latest_metrics.__dict__ if latest_metrics else None,
                'config': self.monitored_strategies[strategy_id]['config'],
                'start_time': self.monitored_strategies[strategy_id]['start_time']
            }
        return overview
    
    def start_monitoring(self) -> None:
        """Start the monitoring loop"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Live strategy monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Live strategy monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update all strategy metrics
                for strategy_id in self.monitored_strategies:
                    self._update_strategy_metrics(strategy_id)
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def export_metrics(self, strategy_id: str, format: str = 'json') -> str:
        """Export strategy metrics for analysis"""
        if strategy_id not in self.strategy_metrics:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        metrics_data = []
        for metric in self.strategy_metrics[strategy_id]:
            metrics_data.append({
                'timestamp': metric.timestamp.isoformat(),
                'pnl': metric.pnl,
                'unrealized_pnl': metric.unrealized_pnl,
                'realized_pnl': metric.realized_pnl,
                'drawdown': metric.current_drawdown,
                'risk_score': metric.risk_score,
                'open_positions': metric.open_positions,
                'win_rate': metric.win_rate
            })
        
        if format.lower() == 'json':
            return json.dumps(metrics_data, indent=2)
        elif format.lower() == 'csv':
            df = pd.DataFrame(metrics_data)
            return df.to_csv(index=False)
        else:
            raise ValueError("Format must be 'json' or 'csv'")


# Example usage and testing
if __name__ == "__main__":
    # Initialize monitor
    monitor = LiveStrategyMonitor({
        'update_interval': 1.0,
        'alert_email': 'trader@example.com'
    })
    
    # Add alert callback
    def alert_handler(alert):
        print(f"ALERT: {alert['level']} - {alert['message']}")
    
    monitor.add_alert_callback(alert_handler)
    
    # Add strategies to monitor
    monitor.add_strategy('scalping_eur_usd', {
        'starting_capital': 50000,
        'risk_limits': {
            'max_drawdown': 0.10,  # 10%
            'max_daily_loss': 0.03,  # 3%
            'max_positions': 5
        }
    })
    
    monitor.add_strategy('swing_trading_gbp_usd', {
        'starting_capital': 100000,
        'risk_limits': {
            'max_drawdown': 0.15,  # 15%
            'max_daily_loss': 0.05,  # 5%
            'max_positions': 10
        }
    })
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some trades
    import random
    for i in range(10):
        strategy = random.choice(['scalping_eur_usd', 'swing_trading_gbp_usd'])
        pnl = random.uniform(-1000, 1500)
        
        monitor.update_trade(strategy, {
            'trade_id': f'trade_{i}',
            'symbol': 'EUR/USD' if 'eur' in strategy else 'GBP/USD',
            'size': random.uniform(0.1, 1.0),
            'pnl': pnl,
            'closed': True
        })
        
        time.sleep(0.1)
    
    # Get strategy overview
    overview = monitor.get_all_strategies()
    print("\nStrategy Overview:")
    for strategy_id, data in overview.items():
        print(f"\n{strategy_id}:")
        print(f"  Status: {data['status']}")
        if data['metrics']:
            print(f"  P&L: ${data['metrics']['pnl']:.2f}")
            print(f"  Drawdown: {data['metrics']['current_drawdown']:.2%}")
            print(f"  Risk Score: {data['metrics']['risk_score']:.1f}")
    
    # Stop monitoring
    time.sleep(2)
    monitor.stop_monitoring()
