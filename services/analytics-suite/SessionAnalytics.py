"""
Advanced Trading Session Analytics System for Platform3

This module provides comprehensive trading session analysis including:
- Session-based performance tracking and comparison
- Market session overlap analysis (London, New York, Tokyo, Sydney)
- Time-zone specific trading patterns and profitability
- Session momentum and market regime detection
- Cross-session strategy performance comparison
- Real-time session monitoring and alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from collections import defaultdict
import pytz
import json
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Platform3 Communication Framework Integration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared', 'communication'))
from platform3_communication_framework import Platform3CommunicationFramework

# Analytics Framework Interface
@dataclass
class RealtimeMetric:
    """Real-time metric data structure"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    alert_threshold: Optional[float] = None

@dataclass
class AnalyticsReport:
    """Standardized analytics report structure"""
    report_id: str
    report_type: str
    generated_at: datetime
    data: Dict[str, Any]
    summary: str
    recommendations: List[str]
    confidence_score: float

class AnalyticsInterface(ABC):
    """Abstract interface for analytics engines"""
    
    @abstractmethod
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data and return analytics results"""
        pass
    
    @abstractmethod
    async def generate_report(self, timeframe: str) -> AnalyticsReport:
        """Generate analytics report for specified timeframe"""
        pass
    
    @abstractmethod
    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        """Get current real-time metrics"""
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSession(Enum):
    """Market trading sessions"""
    SYDNEY = "Sydney"
    TOKYO = "Tokyo"
    LONDON = "London"
    NEW_YORK = "New_York"
    OVERLAP_LONDON_NY = "London_NY_Overlap"
    OVERLAP_TOKYO_LONDON = "Tokyo_London_Overlap"
    WEEKEND = "Weekend"
    
@dataclass
class SessionTrade:
    """Trade with session information"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    duration: float
    session: MarketSession
    session_overlap: bool
    market_volatility: float

@dataclass
class SessionPerformance:
    """Performance metrics for a trading session"""
    session: MarketSession
    start_time: datetime
    end_time: datetime
    total_trades: int
    profitable_trades: int
    win_rate: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_trade_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    trades_per_hour: float
    
@dataclass
class SessionAnalyticsResult:
    """Comprehensive session analytics results"""
    analysis_period: Tuple[datetime, datetime]
    total_sessions_analyzed: int
    session_performances: Dict[MarketSession, SessionPerformance]
    best_performing_session: MarketSession
    worst_performing_session: MarketSession
    most_active_session: MarketSession
    session_overlap_performance: Dict[str, float]
    time_zone_analysis: Dict[str, float]
    session_momentum_patterns: Dict[MarketSession, Dict[str, float]]
    recommendations: List[str]

class SessionAnalytics(AnalyticsInterface):
    """
    Advanced Trading Session Analytics System
    
    Analyzes trading performance across different market sessions,
    time zones, and session overlaps to optimize trading schedules.
    Now implements AnalyticsInterface for framework integration.
    """
    
    def __init__(self, base_timezone: str = 'UTC'):
        """
        Initialize SessionAnalytics
        
        Args:
            base_timezone: Base timezone for analysis (default: UTC)
        """
        self.base_timezone = pytz.timezone(base_timezone)
          # Market session timings (UTC)
        self.session_times = {
            MarketSession.SYDNEY: (21, 6),    # 9 PM - 6 AM UTC
            MarketSession.TOKYO: (0, 9),      # 12 AM - 9 AM UTC  
            MarketSession.LONDON: (8, 17),    # 8 AM - 5 PM UTC
            MarketSession.NEW_YORK: (13, 22)  # 1 PM - 10 PM UTC
        }
        
        # Session overlap periods
        self.overlap_periods = {
            'Tokyo_London': (8, 9),   # 8-9 AM UTC
            'London_NY': (13, 17),    # 1-5 PM UTC
        }
        
        # Performance tracking
        self.session_history = defaultdict(list)
        self.daily_sessions = {}
        
        # Real-time processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._last_update = None
        
        # Platform3 Communication Framework
        self.communication_framework = Platform3CommunicationFramework(
            service_name="session-analytics",
            service_port=8003,
            redis_url="redis://localhost:6379",
            consul_host="localhost",
            consul_port=8500
        )
        
        # Initialize the framework
        try:
            self.communication_framework.initialize()
            logger.info("Session Analytics Communication framework initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize communication framework: {e}")
        
        logger.info(f"SessionAnalytics initialized with timezone: {base_timezone}")
    
    def analyze_session(self, session_data: Union[Dict[str, Any], List[Dict]]) -> SessionAnalyticsResult:
        """
        Analyze trading session performance
        
        Args:
            session_data: Trading data containing trades and timestamps
            
        Returns:
            SessionAnalyticsResult: Comprehensive session analysis
        """
        try:
            # Parse and classify trades by session
            trades = self._parse_trades_with_sessions(session_data)
            
            if not trades:
                logger.warning("No valid trades found in session data")
                return self._create_empty_result()
            
            # Determine analysis period
            analysis_period = self._get_analysis_period(trades)
            
            # Group trades by session
            session_groups = self._group_trades_by_session(trades)
            
            # Calculate performance for each session
            session_performances = {}
            for session, session_trades in session_groups.items():
                if session_trades:
                    performance = self._calculate_session_performance(session, session_trades)
                    session_performances[session] = performance
            
            # Analyze session overlaps
            overlap_performance = self._analyze_session_overlaps(trades)
            
            # Analyze timezone patterns
            timezone_analysis = self._analyze_timezone_patterns(trades)
            
            # Detect session momentum patterns
            momentum_patterns = self._analyze_session_momentum(session_groups)
            
            # Find best/worst sessions
            best_session, worst_session, most_active_session = self._rank_sessions(session_performances)
            
            # Generate recommendations
            recommendations = self._generate_session_recommendations(
                session_performances, overlap_performance, momentum_patterns
            )
            
            result = SessionAnalyticsResult(
                analysis_period=analysis_period,
                total_sessions_analyzed=len(session_performances),
                session_performances=session_performances,
                best_performing_session=best_session,
                worst_performing_session=worst_session,
                most_active_session=most_active_session,
                session_overlap_performance=overlap_performance,
                time_zone_analysis=timezone_analysis,
                session_momentum_patterns=momentum_patterns,
                recommendations=recommendations
            )
            
            # Store for historical analysis
            self._update_session_history(session_performances)
            
            logger.info(f"Session analysis completed: {len(session_performances)} sessions analyzed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in session analysis: {str(e)}")
            return self._create_empty_result()
    
    def _parse_trades_with_sessions(self, session_data: Union[Dict[str, Any], List[Dict]]) -> List[SessionTrade]:
        """Parse trades and classify by market session"""
        trades = []
        
        # Handle different data formats
        if isinstance(session_data, dict):
            trade_list = session_data.get('trades', [])
        elif isinstance(session_data, list):
            trade_list = session_data
        else:
            logger.error("Invalid session data format")
            return trades
        
        for trade_data in trade_list:
            try:
                # Parse timestamp
                timestamp_str = trade_data.get('timestamp') or trade_data.get('entry_time')
                timestamp = self._parse_timestamp(timestamp_str)
                
                # Determine market session
                session = self._determine_market_session(timestamp)
                
                # Check for session overlap
                is_overlap = self._is_session_overlap(timestamp)
                
                # Calculate market volatility (simplified)
                volatility = self._estimate_market_volatility(timestamp, trade_data)
                
                trade = SessionTrade(
                    timestamp=timestamp,
                    symbol=trade_data.get('symbol', 'UNKNOWN'),
                    side=trade_data.get('side', 'long'),
                    entry_price=float(trade_data.get('entry_price', 0)),
                    exit_price=float(trade_data.get('exit_price', 0)),
                    position_size=float(trade_data.get('position_size', 0)),
                    pnl=float(trade_data.get('pnl', 0)),
                    duration=float(trade_data.get('duration', 0)),
                    session=session,
                    session_overlap=is_overlap,
                    market_volatility=volatility
                )
                trades.append(trade)
                
            except Exception as e:
                logger.warning(f"Error parsing trade data: {e}")
                continue
        
        return trades
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string and convert to UTC"""
        if not timestamp_str:
            return datetime.now(timezone.utc)
        
        try:
            # Try different formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%f'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # Fallback to ISO format
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc)
            
        except Exception:
            return datetime.now(timezone.utc)
    
    def _determine_market_session(self, timestamp: datetime) -> MarketSession:
        """Determine which market session a timestamp belongs to"""
        # Convert to UTC if needed
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Check for weekend
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.WEEKEND
        
        # Check session times (UTC)
        for session, (start_hour, end_hour) in self.session_times.items():
            if start_hour > end_hour:  # Crosses midnight
                if hour >= start_hour or hour < end_hour:
                    return session
            else:
                if start_hour <= hour < end_hour:
                    return session
        
        # Default to closest session
        return MarketSession.LONDON
    
    def _is_session_overlap(self, timestamp: datetime) -> bool:
        """Check if timestamp falls in a session overlap period"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        hour = timestamp.hour
        
        for overlap_name, (start_hour, end_hour) in self.overlap_periods.items():
            if start_hour <= hour < end_hour:
                return True
        
        return False
    
    def _estimate_market_volatility(self, timestamp: datetime, trade_data: Dict) -> float:
        """Estimate market volatility at trade time (simplified)"""
        # Use spread or price movement as volatility proxy
        entry_price = float(trade_data.get('entry_price', 100))
        exit_price = float(trade_data.get('exit_price', 100))
        
        if entry_price > 0:
            price_change = abs(exit_price - entry_price) / entry_price
            return price_change
        
        # Default volatility by session
        session = self._determine_market_session(timestamp)
        default_volatilities = {
            MarketSession.LONDON: 0.015,
            MarketSession.NEW_YORK: 0.018,
            MarketSession.TOKYO: 0.012,
            MarketSession.SYDNEY: 0.010,
            MarketSession.WEEKEND: 0.005
        }
        
        return default_volatilities.get(session, 0.012)
    
    def _get_analysis_period(self, trades: List[SessionTrade]) -> Tuple[datetime, datetime]:
        """Get start and end times of analysis period"""
        if not trades:
            now = datetime.now(timezone.utc)
            return (now, now)
        
        timestamps = [trade.timestamp for trade in trades]
        return (min(timestamps), max(timestamps))
    
    def _group_trades_by_session(self, trades: List[SessionTrade]) -> Dict[MarketSession, List[SessionTrade]]:
        """Group trades by market session"""
        session_groups = defaultdict(list)
        
        for trade in trades:
            session_groups[trade.session].append(trade)
        
        return dict(session_groups)
    
    def _calculate_session_performance(self, session: MarketSession, 
                                     trades: List[SessionTrade]) -> SessionPerformance:
        """Calculate performance metrics for a session"""
        if not trades:
            return self._create_empty_session_performance(session)
        
        # Basic metrics
        total_trades = len(trades)
        profitable_trades = sum(1 for trade in trades if trade.pnl > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        pnls = [trade.pnl for trade in trades]
        total_pnl = sum(pnls)
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_trade_pnl = np.mean(pnls)
        
        # Risk metrics
        pnl_array = np.array(pnls)
        volatility = np.std(pnl_array) if len(pnl_array) > 1 else 0
        
        # Sharpe ratio (simplified)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        sharpe_ratio = (avg_trade_pnl - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_pnl = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Trade characteristics
        best_trade = max(pnls)
        worst_trade = min(pnls)
        
        durations = [trade.duration for trade in trades if trade.duration > 0]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Time analysis
        timestamps = [trade.timestamp for trade in trades]
        if len(timestamps) > 1:
            time_span_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            trades_per_hour = total_trades / time_span_hours if time_span_hours > 0 else 0
        else:
            trades_per_hour = 0
        
        return SessionPerformance(
            session=session,
            start_time=min(timestamps),
            end_time=max(timestamps),
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_trade_duration,
            trades_per_hour=trades_per_hour
        )
    
    def _analyze_session_overlaps(self, trades: List[SessionTrade]) -> Dict[str, float]:
        """Analyze performance during session overlaps"""
        overlap_performance = {}
        
        # Group trades by overlap status
        overlap_trades = [trade for trade in trades if trade.session_overlap]
        non_overlap_trades = [trade for trade in trades if not trade.session_overlap]
        
        # Calculate performance metrics
        if overlap_trades:
            overlap_pnl = [trade.pnl for trade in overlap_trades]
            overlap_performance['overlap_avg_pnl'] = np.mean(overlap_pnl)
            overlap_performance['overlap_total_pnl'] = sum(overlap_pnl)
            overlap_performance['overlap_win_rate'] = sum(1 for pnl in overlap_pnl if pnl > 0) / len(overlap_pnl)
            overlap_performance['overlap_trades_count'] = len(overlap_trades)
        
        if non_overlap_trades:
            non_overlap_pnl = [trade.pnl for trade in non_overlap_trades]
            overlap_performance['non_overlap_avg_pnl'] = np.mean(non_overlap_pnl)
            overlap_performance['non_overlap_total_pnl'] = sum(non_overlap_pnl)
            overlap_performance['non_overlap_win_rate'] = sum(1 for pnl in non_overlap_pnl if pnl > 0) / len(non_overlap_pnl)
            overlap_performance['non_overlap_trades_count'] = len(non_overlap_trades)
        
        # Calculate overlap advantage
        if overlap_trades and non_overlap_trades:
            overlap_avg = overlap_performance.get('overlap_avg_pnl', 0)
            non_overlap_avg = overlap_performance.get('non_overlap_avg_pnl', 0)
            overlap_performance['overlap_advantage'] = overlap_avg - non_overlap_avg
        
        return overlap_performance
    
    def _analyze_timezone_patterns(self, trades: List[SessionTrade]) -> Dict[str, float]:
        """Analyze trading patterns by timezone/hour"""
        hourly_performance = defaultdict(list)
        
        for trade in trades:
            hour = trade.timestamp.hour
            hourly_performance[hour].append(trade.pnl)
        
        # Calculate average PnL by hour
        hourly_avg_pnl = {}
        for hour, pnls in hourly_performance.items():
            if len(pnls) >= 2:  # Minimum trades for significance
                hourly_avg_pnl[f'hour_{hour}'] = np.mean(pnls)
        
        # Find best and worst hours
        if hourly_avg_pnl:
            best_hour = max(hourly_avg_pnl.items(), key=lambda x: x[1])
            worst_hour = min(hourly_avg_pnl.items(), key=lambda x: x[1])
            hourly_avg_pnl['best_hour'] = int(best_hour[0].split('_')[1])
            hourly_avg_pnl['best_hour_pnl'] = best_hour[1]
            hourly_avg_pnl['worst_hour'] = int(worst_hour[0].split('_')[1])
            hourly_avg_pnl['worst_hour_pnl'] = worst_hour[1]
        
        return hourly_avg_pnl
    
    def _analyze_session_momentum(self, session_groups: Dict[MarketSession, List[SessionTrade]]) -> Dict[MarketSession, Dict[str, float]]:
        """Analyze momentum patterns within each session"""
        momentum_patterns = {}
        
        for session, trades in session_groups.items():
            if len(trades) < 5:  # Need minimum trades for momentum analysis
                continue
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.timestamp)
            
            # Calculate momentum metrics
            pnls = [trade.pnl for trade in sorted_trades]
            cumulative_pnl = np.cumsum(pnls)
            
            # Early session vs late session performance
            mid_point = len(sorted_trades) // 2
            early_trades = sorted_trades[:mid_point]
            late_trades = sorted_trades[mid_point:]
            
            early_pnl = sum(trade.pnl for trade in early_trades)
            late_pnl = sum(trade.pnl for trade in late_trades)
            
            # Momentum indicators
            momentum_patterns[session] = {
                'early_session_pnl': early_pnl,
                'late_session_pnl': late_pnl,
                'momentum_direction': 'positive' if late_pnl > early_pnl else 'negative',
                'session_trend': np.corrcoef(range(len(pnls)), cumulative_pnl)[0, 1] if len(pnls) > 1 else 0,
                'volatility_trend': np.std(pnls[:mid_point]) - np.std(pnls[mid_point:]) if mid_point > 1 else 0
            }
        
        return momentum_patterns
    
    def _rank_sessions(self, session_performances: Dict[MarketSession, SessionPerformance]) -> Tuple[MarketSession, MarketSession, MarketSession]:
        """Rank sessions by performance metrics"""
        if not session_performances:
            return MarketSession.LONDON, MarketSession.LONDON, MarketSession.LONDON
        
        # Best performing by total PnL
        best_session = max(session_performances.keys(), 
                          key=lambda s: session_performances[s].total_pnl)
        
        # Worst performing by total PnL
        worst_session = min(session_performances.keys(), 
                           key=lambda s: session_performances[s].total_pnl)
        
        # Most active by trade count
        most_active_session = max(session_performances.keys(), 
                                 key=lambda s: session_performances[s].total_trades)
        
        return best_session, worst_session, most_active_session
    
    def _generate_session_recommendations(self, session_performances: Dict[MarketSession, SessionPerformance],
                                        overlap_performance: Dict[str, float],
                                        momentum_patterns: Dict[MarketSession, Dict[str, float]]) -> List[str]:
        """Generate actionable session trading recommendations"""
        recommendations = []
        
        if not session_performances:
            return ["Insufficient data for session recommendations"]
        
        # Session performance recommendations
        best_sessions = sorted(session_performances.items(), 
                             key=lambda x: x[1].total_pnl, reverse=True)[:2]
        
        for session, performance in best_sessions:
            if performance.total_pnl > 0 and performance.win_rate > 0.5:
                recommendations.append(
                    f"Focus trading during {session.value} session: "
                    f"{performance.win_rate:.1%} win rate, ${performance.total_pnl:.2f} total profit"
                )
        
        # Overlap recommendations
        overlap_advantage = overlap_performance.get('overlap_advantage', 0)
        if overlap_advantage > 0:
            recommendations.append(
                f"Trade during session overlaps for ${overlap_advantage:.2f} average advantage per trade"
            )
        
        # Time-based recommendations
        worst_sessions = [s for s, p in session_performances.items() if p.total_pnl < 0]
        if worst_sessions:
            for session in worst_sessions[:2]:
                recommendations.append(f"Avoid trading during {session.value} session due to negative performance")
        
        # Momentum recommendations
        for session, momentum in momentum_patterns.items():
            if momentum['momentum_direction'] == 'positive' and momentum['session_trend'] > 0.5:
                recommendations.append(
                    f"Consider scaling position sizes during {session.value} session due to positive momentum"
                )
        
        return recommendations if recommendations else ["No specific session recommendations available"]
    
    def _create_empty_session_performance(self, session: MarketSession) -> SessionPerformance:
        """Create empty session performance for error cases"""
        now = datetime.now(timezone.utc)
        return SessionPerformance(
            session=session, start_time=now, end_time=now, total_trades=0,
            profitable_trades=0, win_rate=0, total_pnl=0, gross_profit=0,
            gross_loss=0, profit_factor=0, avg_trade_pnl=0, sharpe_ratio=0,
            max_drawdown=0, volatility=0, best_trade=0, worst_trade=0,
            avg_trade_duration=0, trades_per_hour=0
        )
    
    def _create_empty_result(self) -> SessionAnalyticsResult:
        """Create empty result for error cases"""
        now = datetime.now(timezone.utc)
        return SessionAnalyticsResult(
            analysis_period=(now, now), total_sessions_analyzed=0,
            session_performances={}, best_performing_session=MarketSession.LONDON,
            worst_performing_session=MarketSession.LONDON,
            most_active_session=MarketSession.LONDON, session_overlap_performance={},
            time_zone_analysis={}, session_momentum_patterns={}, recommendations=[]
        )
    
    def _update_session_history(self, session_performances: Dict[MarketSession, SessionPerformance]):
        """Update historical session performance tracking"""
        for session, performance in session_performances.items():
            self.session_history[session].append({
                'date': datetime.now(timezone.utc).date(),
                'total_pnl': performance.total_pnl,
                'win_rate': performance.win_rate,
                'total_trades': performance.total_trades,
                'sharpe_ratio': performance.sharpe_ratio
            })
    
    def get_session_comparison_report(self, result: SessionAnalyticsResult) -> str:
        """Generate detailed session comparison report"""
        report = f"""
TRADING SESSION ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {result.analysis_period[0].strftime('%Y-%m-%d')} to {result.analysis_period[1].strftime('%Y-%m-%d')}
===============================================

SESSION PERFORMANCE SUMMARY:
- Total Sessions Analyzed: {result.total_sessions_analyzed}
- Best Performing Session: {result.best_performing_session.value}
- Worst Performing Session: {result.worst_performing_session.value}
- Most Active Session: {result.most_active_session.value}

DETAILED SESSION BREAKDOWN:
"""
        
        for session, performance in result.session_performances.items():
            report += f"""
{session.value.upper()} SESSION:
- Total Trades: {performance.total_trades}
- Win Rate: {performance.win_rate:.1%}
- Total P&L: ${performance.total_pnl:,.2f}
- Profit Factor: {performance.profit_factor:.2f}
- Sharpe Ratio: {performance.sharpe_ratio:.3f}
- Avg Trade Duration: {performance.avg_trade_duration:.1f} minutes
- Trades per Hour: {performance.trades_per_hour:.1f}
"""
        
        # Session overlap analysis
        if result.session_overlap_performance:
            report += "\nSESSION OVERLAP ANALYSIS:\n"
            overlap_data = result.session_overlap_performance
            if 'overlap_advantage' in overlap_data:
                report += f"- Overlap Advantage: ${overlap_data['overlap_advantage']:.2f} per trade\n"
            if 'overlap_win_rate' in overlap_data:
                report += f"- Overlap Win Rate: {overlap_data['overlap_win_rate']:.1%}\n"
        
        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def export_session_data(self, result: SessionAnalyticsResult, filepath: str) -> bool:
        """Export session analysis data to JSON"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_period': [dt.isoformat() for dt in result.analysis_period],
                'session_performances': {},
                'overlap_performance': result.session_overlap_performance,
                'timezone_analysis': result.time_zone_analysis,
                'recommendations': result.recommendations
            }
            
            # Convert session performances to serializable format
            for session, performance in result.session_performances.items():
                export_data['session_performances'][session.value] = {
                    'total_trades': performance.total_trades,
                    'win_rate': performance.win_rate,
                    'total_pnl': performance.total_pnl,
                    'profit_factor': performance.profit_factor,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'trades_per_hour': performance.trades_per_hour
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Session data exported to {filepath}")
            return True
            
        except Exception as e:            logger.error(f"Error exporting session data: {e}")
    return False

    # AnalyticsInterface Implementation for Framework Integration
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming data and return analytics results
        Implements AnalyticsInterface for framework integration
        """
        try:
            # Update last update timestamp
            self._last_update = datetime.now()
            
            # Process session-specific trading data
            if 'session_trades' in data:
                # Process session trade data
                session_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.analyze_session, 
                    data['session_trades']
                )
                
                return {
                    "success": True,
                    "total_sessions": session_results.total_sessions_analyzed,
                    "best_session": session_results.best_performing_session.value,
                    "worst_session": session_results.worst_performing_session.value,
                    "most_active": session_results.most_active_session.value,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif 'market_data' in data:
                # Process market data for session analysis
                market_data = data['market_data']
                current_session = self._get_current_session()
                
                return {
                    "success": True,
                    "current_session": current_session.value if current_session else "NONE",
                    "data_points": len(market_data) if isinstance(market_data, list) else 1,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Return empty result if no processable data
            return {
                "success": False,
                "message": "No processable session data found",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing data in SessionAnalytics: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def generate_report(self, timeframe: str) -> AnalyticsReport:
        """
        Generate analytics report for specified timeframe
        Implements AnalyticsInterface for framework integration
        """
        try:
            # Generate comprehensive session analytics report
            current_session = self._get_current_session()
            
            report_data = {
                "timeframe": timeframe,
                "current_session": current_session.value if current_session else "NONE",
                "session_timings": {
                    session.value: {"start": times[0], "end": times[1]} 
                    for session, times in self.session_times.items()
                },
                "overlap_periods": self.overlap_periods,
                "session_history_count": len(self.session_history),
                "base_timezone": str(self.base_timezone)
            }
            
            # Add session history data if available
            if self.session_history:
                report_data["active_sessions"] = list(self.session_history.keys())
                
            # Generate session-specific recommendations
            recommendations = [
                "Focus trading during high-liquidity overlap periods",
                "Avoid low-volatility weekend and session transition periods",
                "Optimize position sizing based on session characteristics",
                "Monitor timezone-specific economic announcements",
                "Adjust strategies for different session momentum patterns"
            ]
            
            # Calculate confidence score based on session data
            confidence_score = 82.0
            if self.session_history:
                confidence_score = min(94.0, 82.0 + min(len(self.session_history), 40) * 0.3)
            
            summary = f"Session analytics report for {timeframe} showing trading performance across market sessions"
            
            return AnalyticsReport(
                report_id=f"session_analytics_{timeframe}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                report_type="session_analytics",
                generated_at=datetime.utcnow(),
                data=report_data,
                summary=summary,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating session analytics report: {e}")
            return AnalyticsReport(
                report_id=f"session_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                report_type="session_analytics",
                generated_at=datetime.utcnow(),
                data={"error": str(e)},
                summary=f"Error generating session analytics report: {str(e)}",
                recommendations=["Review data input", "Check timezone configuration"],
                confidence_score=0.0
            )

    def get_real_time_metrics(self) -> List[RealtimeMetric]:
        """
        Get current real-time metrics
        Implements AnalyticsInterface for framework integration
        """
        try:
            metrics = []
            current_time = datetime.utcnow()
            
            # Engine status metric
            metrics.append(RealtimeMetric(
                metric_name="session_analytics_engine_status",
                value=1.0,  # 1.0 = active, 0.0 = inactive
                timestamp=current_time,
                context={"engine": "session_analytics", "status": "active"}
            ))
            
            # Current session activity
            current_session = self._get_current_session()
            session_activity = 1.0 if current_session else 0.3
            metrics.append(RealtimeMetric(
                metric_name="current_session_activity",
                value=session_activity,
                timestamp=current_time,
                context={
                    "current_session": current_session.value if current_session else "NONE",
                    "utc_hour": current_time.hour
                }
            ))
            
            # Session overlap detection
            overlap_active = self._is_overlap_period(current_time.hour)
            metrics.append(RealtimeMetric(
                metric_name="session_overlap_active",
                value=1.0 if overlap_active else 0.0,
                timestamp=current_time,
                context={"overlap_detected": overlap_active}
            ))
            
            # Session history utilization
            history_utilization = len(self.session_history) / 50.0  # Normalize
            metrics.append(RealtimeMetric(
                metric_name="session_history_utilization",
                value=min(1.0, history_utilization),
                timestamp=current_time,
                context={"history_count": len(self.session_history)},
                alert_threshold=0.8
            ))
            
            # Processing efficiency metric
            if self._last_update:
                time_since_update = (current_time - self._last_update).total_seconds()
                efficiency = max(0.0, 1.0 - (time_since_update / 3600.0))  # Normalize by hour
                metrics.append(RealtimeMetric(
                    metric_name="session_processing_efficiency",
                    value=efficiency,
                    timestamp=current_time,
                    context={"last_update": self._last_update.isoformat()},
                    alert_threshold=0.3
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting session analytics real-time metrics: {e}")
            return []

    def _get_current_session(self) -> Optional[MarketSession]:
        """Get the current active trading session based on UTC time"""
        try:
            current_hour = datetime.utcnow().hour
            
            for session, (start, end) in self.session_times.items():
                if start <= end:  # Normal session (doesn't cross midnight)
                    if start <= current_hour < end:
                        return session
                else:  # Session crosses midnight (like Sydney)
                    if current_hour >= start or current_hour < end:
                        return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining current session: {e}")
            return None

    def _is_overlap_period(self, hour: int) -> bool:
        """Check if current hour is during a session overlap period"""
        try:
            for period_name, (start, end) in self.overlap_periods.items():
                if start <= hour < end:
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking overlap period: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Example session data
    sample_data = [
        {
            'timestamp': '2024-01-01 09:30:00',  # London session
            'symbol': 'EURUSD',
            'side': 'long',
            'entry_price': 1.1000,
            'exit_price': 1.1020,
            'position_size': 100000,
            'pnl': 200.0,
            'duration': 15.5
        },
        {
            'timestamp': '2024-01-01 15:00:00',  # London-NY overlap
            'symbol': 'GBPUSD',
            'side': 'short',
            'entry_price': 1.2500,
            'exit_price': 1.2480,
            'position_size': 50000,
            'pnl': 100.0,
            'duration': 8.2
        },
        {
            'timestamp': '2024-01-01 20:00:00',  # NY session
            'symbol': 'USDCAD',
            'side': 'long',
            'entry_price': 1.3500,
            'exit_price': 1.3485,
            'position_size': 75000,
            'pnl': -112.5,
            'duration': 25.1
        }
    ]
    
    # Initialize session analyzer
    analyzer = SessionAnalytics()
    
    # Analyze sessions
    result = analyzer.analyze_session(sample_data)
    
    # Print report
    print(analyzer.get_session_comparison_report(result))
    
    # Export data (optional)
    # analyzer.export_session_data(result, 'session_analysis.json')

class SessionAnalytics:
    def __init__(self):
        print("SessionAnalytics initialized")

    def analyze_session(self, session_data):
        # Placeholder for session analytics
        print("Analyzing session data...")
