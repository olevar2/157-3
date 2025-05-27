"""
Day Trading Analytics Module

This module provides comprehensive performance analytics specifically designed for
intraday trading strategies, focusing on M15-H1 timeframes and session-based analysis.
Optimized for day trading performance measurement and optimization.

Features:
- Intraday performance metrics and statistics
- Session-based performance breakdown (Asian/London/NY)
- Real-time P&L tracking and analysis
- Win rate and risk-reward analysis
- Drawdown analysis and recovery tracking
- Performance comparison across different sessions
- Trade timing and execution analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, time, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading session classification"""
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"

class TradeOutcome(Enum):
    """Trade outcome classification"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"

@dataclass
class Trade:
    """Individual trade data structure"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    direction: str  # 'long' or 'short'
    pnl: float
    commission: float
    session: TradingSession
    timeframe: str
    strategy: str
    outcome: TradeOutcome

@dataclass
class SessionPerformance:
    """Session-specific performance metrics"""
    session: TradingSession
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    largest_win: float
    largest_loss: float
    average_trade_duration: timedelta

@dataclass
class DayTradingMetrics:
    """Comprehensive day trading performance metrics"""
    total_trades: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    recovery_factor: float
    average_daily_pnl: float
    best_day: float
    worst_day: float
    consecutive_winning_days: int
    consecutive_losing_days: int
    session_breakdown: Dict[str, SessionPerformance]
    hourly_performance: Dict[int, float]
    monthly_performance: Dict[str, float]

class DayTradingAnalytics:
    """
    Day Trading Performance Analytics
    
    Provides comprehensive analytics for intraday trading performance,
    focusing on session-based analysis and real-time metrics.
    """
    
    def __init__(self, account_balance: float = 10000.0):
        """
        Initialize Day Trading Analytics
        
        Args:
            account_balance: Starting account balance for calculations
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        
        # Trade history
        self.trades: List[Trade] = []
        self.daily_pnl: Dict[str, float] = {}  # Date -> PnL
        self.balance_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = account_balance
        
        logger.info(f"DayTradingAnalytics initialized with balance: ${account_balance:,.2f}")
    
    def _identify_session(self, timestamp: datetime) -> TradingSession:
        """
        Identify trading session based on UTC timestamp
        
        Args:
            timestamp: UTC timestamp
            
        Returns:
            TradingSession enum value
        """
        utc_time = timestamp.time()
        
        # Session times in UTC
        if time(13, 0) <= utc_time <= time(16, 0):  # London-NY overlap
            return TradingSession.OVERLAP_LONDON_NY
        elif time(8, 0) <= utc_time <= time(9, 0):  # Asian-London overlap
            return TradingSession.OVERLAP_ASIAN_LONDON
        elif time(8, 0) <= utc_time < time(13, 0):  # London only
            return TradingSession.LONDON
        elif time(16, 0) < utc_time <= time(22, 0):  # NY only
            return TradingSession.NY
        else:  # Asian session
            return TradingSession.ASIAN
    
    def _classify_trade_outcome(self, pnl: float, commission: float = 0.0) -> TradeOutcome:
        """
        Classify trade outcome based on P&L
        
        Args:
            pnl: Trade profit/loss
            commission: Trade commission
            
        Returns:
            TradeOutcome enum value
        """
        net_pnl = pnl - commission
        
        if net_pnl > 0.01:  # Small threshold for breakeven
            return TradeOutcome.WIN
        elif net_pnl < -0.01:
            return TradeOutcome.LOSS
        else:
            return TradeOutcome.BREAKEVEN
    
    def add_trade(self, 
                  entry_time: datetime,
                  exit_time: datetime,
                  entry_price: float,
                  exit_price: float,
                  quantity: float,
                  direction: str,
                  commission: float = 0.0,
                  timeframe: str = "M15",
                  strategy: str = "day_trading") -> None:
        """
        Add a completed trade to the analytics
        
        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            direction: 'long' or 'short'
            commission: Trade commission
            timeframe: Trading timeframe
            strategy: Strategy name
        """
        try:
            # Calculate P&L
            if direction.lower() == 'long':
                pnl = (exit_price - entry_price) * quantity
            else:  # short
                pnl = (entry_price - exit_price) * quantity
            
            # Identify session
            session = self._identify_session(entry_time)
            
            # Classify outcome
            outcome = self._classify_trade_outcome(pnl, commission)
            
            # Create trade object
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                direction=direction.lower(),
                pnl=pnl,
                commission=commission,
                session=session,
                timeframe=timeframe,
                strategy=strategy,
                outcome=outcome
            )
            
            self.trades.append(trade)
            
            # Update account balance
            net_pnl = pnl - commission
            self.account_balance += net_pnl
            self.balance_history.append((exit_time, self.account_balance))
            
            # Update daily P&L
            date_key = exit_time.strftime('%Y-%m-%d')
            if date_key not in self.daily_pnl:
                self.daily_pnl[date_key] = 0.0
            self.daily_pnl[date_key] += net_pnl
            
            # Update drawdown tracking
            if self.account_balance > self.peak_balance:
                self.peak_balance = self.account_balance
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            logger.debug(f"Trade added: {direction} {quantity} @ {entry_price}->{exit_price}, "
                        f"P&L: ${net_pnl:.2f}, Session: {session.value}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
            raise
    
    def _calculate_session_performance(self, session: TradingSession) -> SessionPerformance:
        """
        Calculate performance metrics for a specific session
        
        Args:
            session: Trading session to analyze
            
        Returns:
            SessionPerformance object
        """
        session_trades = [t for t in self.trades if t.session == session]
        
        if not session_trades:
            return SessionPerformance(
                session=session,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                largest_win=0.0,
                largest_loss=0.0,
                average_trade_duration=timedelta()
            )
        
        # Basic metrics
        total_trades = len(session_trades)
        winning_trades = len([t for t in session_trades if t.outcome == TradeOutcome.WIN])
        losing_trades = len([t for t in session_trades if t.outcome == TradeOutcome.LOSS])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        # P&L metrics
        total_pnl = sum(t.pnl - t.commission for t in session_trades)
        wins = [t.pnl - t.commission for t in session_trades if t.outcome == TradeOutcome.WIN]
        losses = [t.pnl - t.commission for t in session_trades if t.outcome == TradeOutcome.LOSS]
        
        average_win = np.mean(wins) if wins else 0.0
        average_loss = np.mean(losses) if losses else 0.0
        
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in session_trades:
            if trade.outcome == TradeOutcome.WIN:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif trade.outcome == TradeOutcome.LOSS:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Largest win/loss
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0
        
        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() for t in session_trades]
        average_duration_seconds = np.mean(durations) if durations else 0.0
        average_trade_duration = timedelta(seconds=average_duration_seconds)
        
        return SessionPerformance(
            session=session,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=average_trade_duration
        )
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio based on daily returns
        
        Returns:
            Sharpe ratio value
        """
        if len(self.daily_pnl) < 2:
            return 0.0
        
        daily_returns = list(self.daily_pnl.values())
        
        # Convert to percentage returns
        daily_return_pct = [pnl / self.initial_balance for pnl in daily_returns]
        
        mean_return = np.mean(daily_return_pct)
        std_return = np.std(daily_return_pct)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_hourly_performance(self) -> Dict[int, float]:
        """
        Calculate performance breakdown by hour of day
        
        Returns:
            Dictionary mapping hour to total P&L
        """
        hourly_pnl = {}
        
        for trade in self.trades:
            hour = trade.entry_time.hour
            net_pnl = trade.pnl - trade.commission
            
            if hour not in hourly_pnl:
                hourly_pnl[hour] = 0.0
            hourly_pnl[hour] += net_pnl
        
        return hourly_pnl
    
    def _calculate_monthly_performance(self) -> Dict[str, float]:
        """
        Calculate performance breakdown by month
        
        Returns:
            Dictionary mapping month to total P&L
        """
        monthly_pnl = {}
        
        for trade in self.trades:
            month_key = trade.exit_time.strftime('%Y-%m')
            net_pnl = trade.pnl - trade.commission
            
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0.0
            monthly_pnl[month_key] += net_pnl
        
        return monthly_pnl
    
    def calculate_metrics(self) -> DayTradingMetrics:
        """
        Calculate comprehensive day trading performance metrics
        
        Returns:
            DayTradingMetrics object with all performance data
        """
        try:
            if not self.trades:
                logger.warning("No trades available for analysis")
                return self._create_empty_metrics()
            
            # Basic metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.outcome == TradeOutcome.WIN])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
            
            # P&L metrics
            total_pnl = sum(t.pnl - t.commission for t in self.trades)
            wins = [t.pnl - t.commission for t in self.trades if t.outcome == TradeOutcome.WIN]
            losses = [t.pnl - t.commission for t in self.trades if t.outcome == TradeOutcome.LOSS]
            
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            # Risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio()
            recovery_factor = total_pnl / (self.max_drawdown * self.initial_balance) if self.max_drawdown > 0 else 0.0
            
            # Daily metrics
            daily_pnls = list(self.daily_pnl.values())
            average_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0.0
            best_day = max(daily_pnls) if daily_pnls else 0.0
            worst_day = min(daily_pnls) if daily_pnls else 0.0
            
            # Consecutive days
            consecutive_winning_days = 0
            consecutive_losing_days = 0
            current_winning_streak = 0
            current_losing_streak = 0
            
            for pnl in daily_pnls:
                if pnl > 0:
                    current_winning_streak += 1
                    current_losing_streak = 0
                    consecutive_winning_days = max(consecutive_winning_days, current_winning_streak)
                elif pnl < 0:
                    current_losing_streak += 1
                    current_winning_streak = 0
                    consecutive_losing_days = max(consecutive_losing_days, current_losing_streak)
            
            # Session breakdown
            session_breakdown = {}
            for session in TradingSession:
                session_perf = self._calculate_session_performance(session)
                session_breakdown[session.value] = session_perf
            
            # Hourly and monthly performance
            hourly_performance = self._calculate_hourly_performance()
            monthly_performance = self._calculate_monthly_performance()
            
            metrics = DayTradingMetrics(
                total_trades=total_trades,
                total_pnl=total_pnl,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self.max_drawdown * 100,  # Convert to percentage
                recovery_factor=recovery_factor,
                average_daily_pnl=average_daily_pnl,
                best_day=best_day,
                worst_day=worst_day,
                consecutive_winning_days=consecutive_winning_days,
                consecutive_losing_days=consecutive_losing_days,
                session_breakdown=session_breakdown,
                hourly_performance=hourly_performance,
                monthly_performance=monthly_performance
            )
            
            logger.info(f"Day trading metrics calculated: {total_trades} trades, "
                       f"Win rate: {win_rate:.1f}%, Total P&L: ${total_pnl:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating day trading metrics: {str(e)}")
            raise
    
    def _create_empty_metrics(self) -> DayTradingMetrics:
        """Create empty metrics for no-trade scenarios"""
        return DayTradingMetrics(
            total_trades=0,
            total_pnl=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            recovery_factor=0.0,
            average_daily_pnl=0.0,
            best_day=0.0,
            worst_day=0.0,
            consecutive_winning_days=0,
            consecutive_losing_days=0,
            session_breakdown={},
            hourly_performance={},
            monthly_performance={}
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key performance metrics
        
        Returns:
            Dictionary with performance summary
        """
        metrics = self.calculate_metrics()
        
        return {
            "account_balance": self.account_balance,
            "total_return": ((self.account_balance - self.initial_balance) / self.initial_balance) * 100,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "current_drawdown": self.current_drawdown * 100,
            "best_session": max(metrics.session_breakdown.items(), 
                              key=lambda x: x[1].total_pnl)[0] if metrics.session_breakdown else "none",
            "most_profitable_hour": max(metrics.hourly_performance.items(), 
                                      key=lambda x: x[1])[0] if metrics.hourly_performance else 0
        }
    
    def get_trading_recommendations(self) -> Dict[str, Any]:
        """
        Generate trading recommendations based on performance analysis
        
        Returns:
            Dictionary with trading recommendations
        """
        metrics = self.calculate_metrics()
        recommendations = {
            "overall_performance": "good" if metrics.win_rate > 60 else "needs_improvement",
            "recommended_sessions": [],
            "risk_adjustments": [],
            "strategy_suggestions": []
        }
        
        # Session recommendations
        if metrics.session_breakdown:
            best_sessions = sorted(metrics.session_breakdown.items(), 
                                 key=lambda x: x[1].total_pnl, reverse=True)[:2]
            recommendations["recommended_sessions"] = [session[0] for session in best_sessions]
        
        # Risk recommendations
        if metrics.max_drawdown > 20:
            recommendations["risk_adjustments"].append("Reduce position sizes")
        if metrics.profit_factor < 1.2:
            recommendations["risk_adjustments"].append("Improve risk-reward ratio")
        
        # Strategy suggestions
        if metrics.win_rate < 50:
            recommendations["strategy_suggestions"].append("Focus on trade selection quality")
        if metrics.sharpe_ratio < 1.0:
            recommendations["strategy_suggestions"].append("Improve consistency of returns")
        
        return recommendations
