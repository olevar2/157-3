"""
Session Performance Tracker

Comprehensive session-based performance tracking for trading strategies with 
detailed analytics, session comparison, and performance optimization insights.

Features:
- Real-time session performance monitoring
- Multi-timeframe session analysis (daily, weekly, monthly)
- Session-based risk metrics and statistics
- Performance pattern recognition
- Trading time optimization
- Session comparison and ranking
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pytz
from concurrent.futures import ThreadPoolExecutor


class SessionType(Enum):
    """Trading session types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class MarketSession(Enum):
    """Market session periods"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OVERLAP_ASIAN_LONDON = "asian_london_overlap"


class PerformanceMetric(Enum):
    """Performance metrics for tracking"""
    PNL = "pnl"
    RETURN = "return"
    SHARPE = "sharpe"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"


@dataclass
class SessionMetrics:
    """Session performance metrics"""
    session_id: str
    session_type: SessionType
    start_time: datetime
    end_time: datetime
    duration_hours: float
    
    # Performance metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    return_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Market exposure
    time_in_market: float
    max_position_size: float
    avg_position_size: float
    
    # Trading efficiency
    trades_per_hour: float
    pnl_per_trade: float
    pnl_per_hour: float
    
    # Session context
    market_session: Optional[MarketSession] = None
    instruments_traded: List[str] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    
    # Additional data
    trade_details: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


@dataclass
class SessionComparison:
    """Session comparison analysis"""
    session_ids: List[str]
    comparison_metric: PerformanceMetric
    best_session: str
    worst_session: str
    average_performance: float
    performance_std: float
    consistency_score: float
    trend_analysis: Dict[str, Any]
    statistical_significance: Dict[str, float]


class SessionPerformanceTracker:
    """
    Advanced session-based performance tracking system
    
    Provides comprehensive analysis of trading sessions with:
    - Real-time session monitoring
    - Multi-timeframe session analysis
    - Performance pattern recognition
    - Trading time optimization
    - Session comparison and ranking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SessionPerformanceTracker
        
        Args:
            config: Configuration dictionary with tracking settings
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Session storage
        self.sessions: Dict[str, SessionMetrics] = {}
        self.active_sessions: Dict[str, Dict] = {}
        self.session_trade_data: Dict[str, List[Dict]] = defaultdict(list)
        
        # Configuration
        self.timezone = pytz.timezone(self.config.get('timezone', 'UTC'))
        self.starting_capital = self.config.get('starting_capital', 100000)
        self.session_types = self.config.get('session_types', [SessionType.DAILY])
        
        # Market session definitions (UTC times)
        self.market_sessions = {
            MarketSession.SYDNEY: (time(21, 0), time(6, 0)),
            MarketSession.ASIAN: (time(23, 0), time(8, 0)),
            MarketSession.LONDON: (time(7, 0), time(16, 0)),
            MarketSession.NEW_YORK: (time(12, 0), time(21, 0)),
            MarketSession.OVERLAP_ASIAN_LONDON: (time(7, 0), time(8, 0)),
            MarketSession.OVERLAP_LONDON_NY: (time(12, 0), time(16, 0))
        }
        
        # Performance analytics
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.session_patterns: Dict[str, Dict] = {}
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("SessionPerformanceTracker initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SessionPerformanceTracker')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_session(self, session_id: str, session_type: SessionType = SessionType.DAILY,
                     start_time: Optional[datetime] = None, strategies: List[str] = None) -> bool:
        """
        Start a new tracking session
        
        Args:
            session_id: Unique session identifier
            session_type: Type of session (daily, weekly, etc.)
            start_time: Session start time (defaults to now)
            strategies: List of strategies to track in this session
            
        Returns:
            bool: True if session started successfully
        """
        try:
            if session_id in self.active_sessions:
                self.logger.warning(f"Session {session_id} already active")
                return False
            
            start_time = start_time or datetime.now(self.timezone)
            
            self.active_sessions[session_id] = {
                'session_type': session_type,
                'start_time': start_time,
                'strategies': strategies or [],
                'starting_equity': self.starting_capital,
                'current_equity': self.starting_capital,
                'trade_count': 0,
                'instruments': set(),
                'market_session': self._identify_market_session(start_time.time())
            }
            
            self.logger.info(f"Started {session_type.value} session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting session {session_id}: {e}")
            return False
    
    def track_performance(self, session_id: str, trade_data: Dict = None, 
                         price_update: Dict = None) -> None:
        """
        Track performance for active session
        
        Args:
            session_id: Session identifier
            trade_data: Trade information (entry/exit, P&L, etc.)
            price_update: Price update for unrealized P&L calculation
        """
        try:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Session {session_id} not active")
                return
            
            session_info = self.active_sessions[session_id]
            
            # Process trade data
            if trade_data:
                self._process_trade_data(session_id, trade_data)
            
            # Process price updates
            if price_update:
                self._process_price_update(session_id, price_update)
            
            # Update real-time metrics
            self._update_session_metrics(session_id)
            
        except Exception as e:
            self.logger.error(f"Error tracking performance for {session_id}: {e}")
    
    def end_session(self, session_id: str, end_time: Optional[datetime] = None) -> Optional[SessionMetrics]:
        """
        End a tracking session and calculate final metrics
        
        Args:
            session_id: Session identifier
            end_time: Session end time (defaults to now)
            
        Returns:
            SessionMetrics: Final session performance metrics
        """
        try:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Session {session_id} not active")
                return None
            
            end_time = end_time or datetime.now(self.timezone)
            session_info = self.active_sessions[session_id]
            
            # Calculate final metrics
            metrics = self._calculate_session_metrics(session_id, end_time)
            
            # Store session
            self.sessions[session_id] = metrics
            
            # Clean up active session
            del self.active_sessions[session_id]
            
            # Update performance history
            self._update_performance_history(metrics)
            
            # Analyze patterns
            self._analyze_session_patterns(metrics)
            
            self.logger.info(f"Ended session {session_id}: P&L ${metrics.total_pnl:.2f}, "
                           f"Win Rate {metrics.win_rate:.1%}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error ending session {session_id}: {e}")
            return None
    
    def _process_trade_data(self, session_id: str, trade_data: Dict) -> None:
        """Process individual trade data"""
        trade_data['timestamp'] = trade_data.get('timestamp', datetime.now(self.timezone))
        trade_data['session_id'] = session_id
        
        # Store trade data
        self.session_trade_data[session_id].append(trade_data)
        
        # Update session info
        session_info = self.active_sessions[session_id]
        session_info['trade_count'] += 1
        
        if 'instrument' in trade_data:
            session_info['instruments'].add(trade_data['instrument'])
        
        # Update equity if trade is closed
        if trade_data.get('closed', False) and 'pnl' in trade_data:
            session_info['current_equity'] += trade_data['pnl']
    
    def _process_price_update(self, session_id: str, price_update: Dict) -> None:
        """Process price updates for unrealized P&L"""
        # Calculate unrealized P&L for open positions
        # This would integrate with position management system
        pass
    
    def _update_session_metrics(self, session_id: str) -> None:
        """Update real-time session metrics"""
        if session_id not in self.active_sessions:
            return
        
        session_info = self.active_sessions[session_id]
        trades = self.session_trade_data[session_id]
        
        if not trades:
            return
        
        # Calculate current performance
        closed_trades = [t for t in trades if t.get('closed', False)]
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        
        # Update current equity
        session_info['current_equity'] = session_info['starting_equity'] + total_pnl
        
        # Store performance snapshot
        current_time = datetime.now(self.timezone)
        self.performance_history[session_id].append({
            'timestamp': current_time,
            'equity': session_info['current_equity'],
            'pnl': total_pnl,
            'trade_count': len(closed_trades)
        })
    
    def _calculate_session_metrics(self, session_id: str, end_time: datetime) -> SessionMetrics:
        """Calculate comprehensive session metrics"""
        session_info = self.active_sessions[session_id]
        trades = self.session_trade_data[session_id]
        
        start_time = session_info['start_time']
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        # Trade analysis
        closed_trades = [t for t in trades if t.get('closed', False)]
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        # P&L calculations
        realized_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        unrealized_pnl = sum(t.get('unrealized_pnl', 0) for t in trades if not t.get('closed', True))
        total_pnl = realized_pnl + unrealized_pnl
        
        # Performance ratios
        starting_equity = session_info['starting_equity']
        return_pct = total_pnl / starting_equity if starting_equity > 0 else 0
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        equity_curve = self._build_equity_curve(session_id)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe and Sortino ratios
        risk_free_rate = self.config.get('risk_free_rate', 0.02)
        excess_return = return_pct * 252 - risk_free_rate  # Annualized
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (return_pct * 252) / max_drawdown if max_drawdown > 0 else 0
        
        # Position analysis
        position_sizes = [abs(t.get('size', 0)) for t in trades if not t.get('closed', True)]
        max_position_size = max(position_sizes) if position_sizes else 0
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        
        # Trading efficiency
        trades_per_hour = len(closed_trades) / duration_hours if duration_hours > 0 else 0
        pnl_per_trade = realized_pnl / len(closed_trades) if closed_trades else 0
        pnl_per_hour = realized_pnl / duration_hours if duration_hours > 0 else 0
        
        # Time in market calculation
        time_in_market = self._calculate_time_in_market(trades, start_time, end_time)
        
        return SessionMetrics(
            session_id=session_id,
            session_type=session_info['session_type'],
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            return_pct=return_pct,
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            time_in_market=time_in_market,
            max_position_size=max_position_size,
            avg_position_size=avg_position_size,
            trades_per_hour=trades_per_hour,
            pnl_per_trade=pnl_per_trade,
            pnl_per_hour=pnl_per_hour,
            market_session=session_info.get('market_session'),
            instruments_traded=list(session_info['instruments']),
            strategies_used=session_info['strategies'],
            trade_details=trades,
            equity_curve=equity_curve,
            timestamps=[h['timestamp'] for h in self.performance_history[session_id]]
        )
    
    def _build_equity_curve(self, session_id: str) -> List[float]:
        """Build equity curve from performance history"""
        history = list(self.performance_history[session_id])
        if not history:
            return [self.active_sessions[session_id]['starting_equity']]
        
        return [h['equity'] for h in history]
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(equity_curve) < 2:
            return 0.0
        
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        
        return abs(drawdown.min())
    
    def _calculate_time_in_market(self, trades: List[Dict], start_time: datetime, end_time: datetime) -> float:
        """Calculate percentage of time with open positions"""
        if not trades:
            return 0.0
        
        total_duration = (end_time - start_time).total_seconds()
        position_time = 0
        
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry = pd.to_datetime(trade['entry_time'])
                exit = pd.to_datetime(trade['exit_time'])
                position_time += (exit - entry).total_seconds()
        
        return position_time / total_duration if total_duration > 0 else 0
    
    def _identify_market_session(self, trade_time: time) -> Optional[MarketSession]:
        """Identify which market session the trade occurred in"""
        for session, (start, end) in self.market_sessions.items():
            if start <= end:  # Same day
                if start <= trade_time <= end:
                    return session
            else:  # Crosses midnight
                if trade_time >= start or trade_time <= end:
                    return session
        
        return None
    
    def _update_performance_history(self, metrics: SessionMetrics) -> None:
        """Update historical performance tracking"""
        session_type = metrics.session_type.value
        
        if session_type not in self.session_patterns:
            self.session_patterns[session_type] = {
                'total_sessions': 0,
                'winning_sessions': 0,
                'avg_pnl': 0,
                'avg_return': 0,
                'best_session': None,
                'worst_session': None,
                'consistency_metrics': {}
            }
        
        patterns = self.session_patterns[session_type]
        patterns['total_sessions'] += 1
        
        if metrics.total_pnl > 0:
            patterns['winning_sessions'] += 1
        
        # Update averages
        patterns['avg_pnl'] = (patterns['avg_pnl'] * (patterns['total_sessions'] - 1) + metrics.total_pnl) / patterns['total_sessions']
        patterns['avg_return'] = (patterns['avg_return'] * (patterns['total_sessions'] - 1) + metrics.return_pct) / patterns['total_sessions']
        
        # Update best/worst
        if patterns['best_session'] is None or metrics.total_pnl > self.sessions[patterns['best_session']].total_pnl:
            patterns['best_session'] = metrics.session_id
        
        if patterns['worst_session'] is None or metrics.total_pnl < self.sessions[patterns['worst_session']].total_pnl:
            patterns['worst_session'] = metrics.session_id
    
    def _analyze_session_patterns(self, metrics: SessionMetrics) -> None:
        """Analyze patterns in session performance"""
        # Market session analysis
        if metrics.market_session:
            session_key = f"market_{metrics.market_session.value}"
            if session_key not in self.session_patterns:
                self.session_patterns[session_key] = {'sessions': [], 'avg_performance': 0}
            
            self.session_patterns[session_key]['sessions'].append(metrics.total_pnl)
            self.session_patterns[session_key]['avg_performance'] = np.mean(self.session_patterns[session_key]['sessions'])
        
        # Day of week analysis
        day_of_week = metrics.start_time.strftime('%A')
        day_key = f"day_{day_of_week.lower()}"
        if day_key not in self.session_patterns:
            self.session_patterns[day_key] = {'sessions': [], 'avg_performance': 0}
        
        self.session_patterns[day_key]['sessions'].append(metrics.total_pnl)
        self.session_patterns[day_key]['avg_performance'] = np.mean(self.session_patterns[day_key]['sessions'])
    
    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get metrics for a completed session"""
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get information about currently active sessions"""
        return self.active_sessions.copy()
    
    def compare_sessions(self, session_ids: List[str], 
                        metric: PerformanceMetric = PerformanceMetric.PNL) -> SessionComparison:
        """
        Compare performance across multiple sessions
        
        Args:
            session_ids: List of session IDs to compare
            metric: Performance metric to use for comparison
            
        Returns:
            SessionComparison: Detailed comparison analysis
        """
        try:
            valid_sessions = [sid for sid in session_ids if sid in self.sessions]
            
            if len(valid_sessions) < 2:
                raise ValueError("Need at least 2 valid sessions for comparison")
            
            # Extract metric values
            metric_values = []
            for session_id in valid_sessions:
                session = self.sessions[session_id]
                
                if metric == PerformanceMetric.PNL:
                    value = session.total_pnl
                elif metric == PerformanceMetric.RETURN:
                    value = session.return_pct
                elif metric == PerformanceMetric.SHARPE:
                    value = session.sharpe_ratio
                elif metric == PerformanceMetric.WIN_RATE:
                    value = session.win_rate
                elif metric == PerformanceMetric.PROFIT_FACTOR:
                    value = session.profit_factor
                elif metric == PerformanceMetric.MAX_DRAWDOWN:
                    value = session.max_drawdown
                elif metric == PerformanceMetric.VOLATILITY:
                    value = session.volatility
                else:
                    value = session.total_pnl
                
                metric_values.append(value)
            
            # Statistical analysis
            performance_array = np.array(metric_values)
            avg_performance = np.mean(performance_array)
            performance_std = np.std(performance_array)
            
            # Find best and worst
            best_idx = np.argmax(performance_array)
            worst_idx = np.argmin(performance_array)
            best_session = valid_sessions[best_idx]
            worst_session = valid_sessions[worst_idx]
            
            # Consistency score (lower std relative to mean = more consistent)
            consistency_score = 100 * (1 - (performance_std / abs(avg_performance))) if avg_performance != 0 else 0
            consistency_score = max(0, min(100, consistency_score))
            
            # Trend analysis
            trend_analysis = self._analyze_performance_trend(valid_sessions, metric_values)
            
            # Statistical significance tests
            statistical_significance = self._calculate_statistical_significance(metric_values)
            
            return SessionComparison(
                session_ids=valid_sessions,
                comparison_metric=metric,
                best_session=best_session,
                worst_session=worst_session,
                average_performance=avg_performance,
                performance_std=performance_std,
                consistency_score=consistency_score,
                trend_analysis=trend_analysis,
                statistical_significance=statistical_significance
            )
            
        except Exception as e:
            self.logger.error(f"Error comparing sessions: {e}")
            raise
    
    def _analyze_performance_trend(self, session_ids: List[str], metric_values: List[float]) -> Dict[str, Any]:
        """Analyze trend in performance over time"""
        if len(metric_values) < 3:
            return {'trend': 'insufficient_data'}
        
        # Linear regression to find trend
        x = np.arange(len(metric_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, metric_values)
        
        # Determine trend direction
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                trend = 'improving'
            else:
                trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'significance': 'significant' if p_value < 0.05 else 'not_significant'
        }
    
    def _calculate_statistical_significance(self, metric_values: List[float]) -> Dict[str, float]:
        """Calculate statistical significance of performance differences"""
        if len(metric_values) < 2:
            return {}
        
        # Test for normality
        _, normality_p = stats.shapiro(metric_values) if len(metric_values) <= 5000 else stats.jarque_bera(metric_values)
        
        # One-sample t-test against zero
        t_stat, t_p = stats.ttest_1samp(metric_values, 0)
        
        return {
            'normality_p_value': normality_p,
            'is_normal': normality_p > 0.05,
            't_statistic': t_stat,
            't_test_p_value': t_p,
            'significantly_different_from_zero': t_p < 0.05
        }
    
    def get_performance_summary(self, session_type: Optional[SessionType] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get performance summary for sessions
        
        Args:
            session_type: Filter by session type
            start_date: Filter sessions after this date
            end_date: Filter sessions before this date
            
        Returns:
            Dict: Performance summary statistics
        """
        # Filter sessions
        filtered_sessions = []
        for session in self.sessions.values():
            if session_type and session.session_type != session_type:
                continue
            if start_date and session.start_time < start_date:
                continue
            if end_date and session.end_time > end_date:
                continue
            filtered_sessions.append(session)
        
        if not filtered_sessions:
            return {'error': 'No sessions found matching criteria'}
        
        # Calculate summary statistics
        total_pnl = sum(s.total_pnl for s in filtered_sessions)
        total_trades = sum(s.total_trades for s in filtered_sessions)
        winning_sessions = sum(1 for s in filtered_sessions if s.total_pnl > 0)
        
        pnl_values = [s.total_pnl for s in filtered_sessions]
        return_values = [s.return_pct for s in filtered_sessions]
        
        summary = {
            'total_sessions': len(filtered_sessions),
            'winning_sessions': winning_sessions,
            'session_win_rate': winning_sessions / len(filtered_sessions),
            'total_pnl': total_pnl,
            'average_pnl': np.mean(pnl_values),
            'median_pnl': np.median(pnl_values),
            'pnl_std': np.std(pnl_values),
            'total_trades': total_trades,
            'average_trades_per_session': total_trades / len(filtered_sessions),
            'average_return': np.mean(return_values),
            'best_session': max(filtered_sessions, key=lambda s: s.total_pnl).session_id,
            'worst_session': min(filtered_sessions, key=lambda s: s.total_pnl).session_id,
            'most_consistent_metric': self._find_most_consistent_metric(filtered_sessions)
        }
        
        return summary
    
    def _find_most_consistent_metric(self, sessions: List[SessionMetrics]) -> str:
        """Find the most consistent performance metric across sessions"""
        metrics = {
            'pnl': [s.total_pnl for s in sessions],
            'return': [s.return_pct for s in sessions],
            'win_rate': [s.win_rate for s in sessions],
            'sharpe': [s.sharpe_ratio for s in sessions],
            'trades_per_hour': [s.trades_per_hour for s in sessions]
        }
        
        consistency_scores = {}
        for metric_name, values in metrics.items():
            if len(values) > 1:
                cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else float('inf')
                consistency_scores[metric_name] = cv
        
        return min(consistency_scores, key=consistency_scores.get) if consistency_scores else 'none'
    
    def export_session_data(self, session_id: str, format: str = 'json') -> str:
        """Export session data for external analysis"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if format.lower() == 'json':
            export_data = {
                'session_id': session.session_id,
                'session_type': session.session_type.value,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat(),
                'duration_hours': session.duration_hours,
                'performance': {
                    'total_pnl': session.total_pnl,
                    'return_pct': session.return_pct,
                    'win_rate': session.win_rate,
                    'sharpe_ratio': session.sharpe_ratio,
                    'max_drawdown': session.max_drawdown
                },
                'trades': session.trade_details,
                'equity_curve': session.equity_curve
            }
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == 'csv':
            df = pd.DataFrame(session.trade_details)
            return df.to_csv(index=False)
        
        else:
            raise ValueError("Format must be 'json' or 'csv'")


# Example usage and testing
if __name__ == "__main__":
    # Initialize tracker
    tracker = SessionPerformanceTracker({
        'timezone': 'UTC',
        'starting_capital': 100000
    })
    
    # Start a daily session
    session_id = 'daily_session_2024_01_15'
    tracker.start_session(session_id, SessionType.DAILY, strategies=['scalping', 'swing'])
    
    # Simulate some trades
    import random
    for i in range(10):
        trade_data = {
            'trade_id': f'trade_{i}',
            'instrument': random.choice(['EUR/USD', 'GBP/USD', 'USD/JPY']),
            'size': random.uniform(0.1, 1.0),
            'pnl': random.uniform(-500, 800),
            'closed': True,
            'entry_time': datetime.now() - timedelta(hours=random.uniform(0, 8)),
            'exit_time': datetime.now() - timedelta(hours=random.uniform(0, 4))
        }
        tracker.track_performance(session_id, trade_data=trade_data)
    
    # End session and get metrics
    metrics = tracker.end_session(session_id)
    
    if metrics:
        print(f"Session Performance Summary:")
        print(f"Total P&L: ${metrics.total_pnl:.2f}")
        print(f"Return: {metrics.return_pct:.2%}")
        print(f"Win Rate: {metrics.win_rate:.1%}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Trades per Hour: {metrics.trades_per_hour:.1f}")
        print(f"Market Session: {metrics.market_session.value if metrics.market_session else 'Unknown'}")
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    print(f"\nOverall Summary:")
    print(f"Total Sessions: {summary['total_sessions']}")
    print(f"Session Win Rate: {summary['session_win_rate']:.1%}")
    print(f"Average P&L: ${summary['average_pnl']:.2f}")
