"""
Swing Trading Analytics Module

This module provides comprehensive performance analytics specifically designed for
swing trading strategies, focusing on H1-H4 timeframes and multi-day position analysis.
Optimized for swing trading performance measurement and optimization.

Features:
- Multi-day position tracking and analysis
- Swing pattern recognition performance
- Hold time analysis and optimization
- Risk-adjusted return metrics
- Market regime performance analysis
- Fibonacci and support/resistance level analysis
- Trend following vs counter-trend performance

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwingType(Enum):
    """Swing trading pattern types"""
    TREND_FOLLOWING = "trend_following"
    COUNTER_TREND = "counter_trend"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    SUPPORT_RESISTANCE = "support_resistance"

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

class HoldPeriod(Enum):
    """Hold period classification"""
    INTRADAY = "intraday"          # < 1 day
    SHORT_SWING = "short_swing"    # 1-3 days
    MEDIUM_SWING = "medium_swing"  # 3-7 days
    LONG_SWING = "long_swing"      # > 7 days

@dataclass
class SwingTrade:
    """Swing trade data structure"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    direction: str
    pnl: float
    commission: float
    swing_type: SwingType
    market_regime: MarketRegime
    hold_period: HoldPeriod
    max_favorable_excursion: float  # MFE
    max_adverse_excursion: float    # MAE
    entry_pattern: str
    exit_reason: str
    risk_reward_ratio: float

@dataclass
class SwingPerformanceMetrics:
    """Comprehensive swing trading performance metrics"""
    total_trades: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    average_hold_time: timedelta
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    recovery_factor: float
    calmar_ratio: float
    sortino_ratio: float
    swing_type_performance: Dict[str, Dict[str, float]]
    regime_performance: Dict[str, Dict[str, float]]
    hold_period_analysis: Dict[str, Dict[str, float]]
    monthly_performance: Dict[str, float]
    risk_metrics: Dict[str, float]

class SwingAnalytics:
    """
    Swing Trading Performance Analytics
    
    Provides comprehensive analytics for swing trading performance,
    focusing on multi-day position analysis and pattern recognition.
    """
    
    def __init__(self, account_balance: float = 50000.0):
        """
        Initialize Swing Trading Analytics
        
        Args:
            account_balance: Starting account balance for calculations
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        
        # Trade history
        self.swing_trades: List[SwingTrade] = []
        self.balance_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.peak_balance = account_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        logger.info(f"SwingAnalytics initialized with balance: ${account_balance:,.2f}")
    
    def _classify_hold_period(self, entry_time: datetime, exit_time: datetime) -> HoldPeriod:
        """
        Classify hold period based on trade duration
        
        Args:
            entry_time: Trade entry time
            exit_time: Trade exit time
            
        Returns:
            HoldPeriod classification
        """
        duration = exit_time - entry_time
        days = duration.total_seconds() / (24 * 3600)
        
        if days < 1:
            return HoldPeriod.INTRADAY
        elif days <= 3:
            return HoldPeriod.SHORT_SWING
        elif days <= 7:
            return HoldPeriod.MEDIUM_SWING
        else:
            return HoldPeriod.LONG_SWING
    
    def _calculate_risk_reward_ratio(self, entry_price: float, exit_price: float,
                                   stop_loss: float, direction: str) -> float:
        """
        Calculate risk-reward ratio for the trade
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss level
            direction: Trade direction
            
        Returns:
            Risk-reward ratio
        """
        if direction.lower() == 'long':
            risk = abs(entry_price - stop_loss)
            reward = abs(exit_price - entry_price)
        else:  # short
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - exit_price)
        
        return reward / risk if risk > 0 else 0.0
    
    def add_swing_trade(self,
                       entry_time: datetime,
                       exit_time: datetime,
                       entry_price: float,
                       exit_price: float,
                       quantity: float,
                       direction: str,
                       swing_type: SwingType,
                       market_regime: MarketRegime,
                       commission: float = 0.0,
                       max_favorable_excursion: float = 0.0,
                       max_adverse_excursion: float = 0.0,
                       entry_pattern: str = "unknown",
                       exit_reason: str = "target",
                       stop_loss: Optional[float] = None) -> None:
        """
        Add a completed swing trade to the analytics
        
        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            direction: 'long' or 'short'
            swing_type: Type of swing pattern
            market_regime: Market regime during trade
            commission: Trade commission
            max_favorable_excursion: Maximum favorable price movement
            max_adverse_excursion: Maximum adverse price movement
            entry_pattern: Entry pattern description
            exit_reason: Reason for exit
            stop_loss: Stop loss level (for R:R calculation)
        """
        try:
            # Calculate P&L
            if direction.lower() == 'long':
                pnl = (exit_price - entry_price) * quantity
            else:  # short
                pnl = (entry_price - exit_price) * quantity
            
            # Classify hold period
            hold_period = self._classify_hold_period(entry_time, exit_time)
            
            # Calculate risk-reward ratio
            risk_reward_ratio = 0.0
            if stop_loss is not None:
                risk_reward_ratio = self._calculate_risk_reward_ratio(
                    entry_price, exit_price, stop_loss, direction
                )
            
            # Create swing trade object
            swing_trade = SwingTrade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                direction=direction.lower(),
                pnl=pnl,
                commission=commission,
                swing_type=swing_type,
                market_regime=market_regime,
                hold_period=hold_period,
                max_favorable_excursion=max_favorable_excursion,
                max_adverse_excursion=max_adverse_excursion,
                entry_pattern=entry_pattern,
                exit_reason=exit_reason,
                risk_reward_ratio=risk_reward_ratio
            )
            
            self.swing_trades.append(swing_trade)
            
            # Update account balance
            net_pnl = pnl - commission
            self.account_balance += net_pnl
            self.balance_history.append((exit_time, self.account_balance))
            
            # Update drawdown tracking
            if self.account_balance > self.peak_balance:
                self.peak_balance = self.account_balance
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            logger.debug(f"Swing trade added: {direction} {swing_type.value}, "
                        f"Hold: {hold_period.value}, P&L: ${net_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding swing trade: {str(e)}")
            raise
    
    def _analyze_swing_type_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by swing type
        
        Returns:
            Dictionary with performance metrics by swing type
        """
        swing_type_performance = {}
        
        for swing_type in SwingType:
            type_trades = [t for t in self.swing_trades if t.swing_type == swing_type]
            
            if not type_trades:
                continue
            
            total_trades = len(type_trades)
            winning_trades = len([t for t in type_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            total_pnl = sum(t.pnl - t.commission for t in type_trades)
            average_pnl = total_pnl / total_trades
            
            wins = [t.pnl - t.commission for t in type_trades if t.pnl > 0]
            losses = [t.pnl - t.commission for t in type_trades if t.pnl < 0]
            
            average_win = np.mean(wins) if wins else 0.0
            average_loss = np.mean(losses) if losses else 0.0
            
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf') if wins else 0.0
            
            # Average hold time
            hold_times = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in type_trades]
            average_hold_hours = np.mean(hold_times)
            
            swing_type_performance[swing_type.value] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_pnl': average_pnl,
                'average_win': average_win,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'average_hold_hours': average_hold_hours
            }
        
        return swing_type_performance
    
    def _analyze_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by market regime
        
        Returns:
            Dictionary with performance metrics by market regime
        """
        regime_performance = {}
        
        for regime in MarketRegime:
            regime_trades = [t for t in self.swing_trades if t.market_regime == regime]
            
            if not regime_trades:
                continue
            
            total_trades = len(regime_trades)
            winning_trades = len([t for t in regime_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            total_pnl = sum(t.pnl - t.commission for t in regime_trades)
            average_pnl = total_pnl / total_trades
            
            wins = [t.pnl - t.commission for t in regime_trades if t.pnl > 0]
            losses = [t.pnl - t.commission for t in regime_trades if t.pnl < 0]
            
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf') if wins else 0.0
            
            regime_performance[regime.value] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_pnl': average_pnl,
                'profit_factor': profit_factor
            }
        
        return regime_performance
    
    def _analyze_hold_period_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by hold period
        
        Returns:
            Dictionary with performance metrics by hold period
        """
        hold_period_performance = {}
        
        for hold_period in HoldPeriod:
            period_trades = [t for t in self.swing_trades if t.hold_period == hold_period]
            
            if not period_trades:
                continue
            
            total_trades = len(period_trades)
            winning_trades = len([t for t in period_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            total_pnl = sum(t.pnl - t.commission for t in period_trades)
            average_pnl = total_pnl / total_trades
            
            wins = [t.pnl - t.commission for t in period_trades if t.pnl > 0]
            losses = [t.pnl - t.commission for t in period_trades if t.pnl < 0]
            
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf') if wins else 0.0
            
            # Average hold time for this category
            hold_times = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in period_trades]
            average_hold_hours = np.mean(hold_times)
            
            hold_period_performance[hold_period.value] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_pnl': average_pnl,
                'profit_factor': profit_factor,
                'average_hold_hours': average_hold_hours
            }
        
        return hold_period_performance
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Returns:
            Dictionary with risk metrics
        """
        if not self.swing_trades:
            return {}
        
        # Daily returns for risk calculations
        daily_returns = []
        if len(self.balance_history) > 1:
            for i in range(1, len(self.balance_history)):
                prev_balance = self.balance_history[i-1][1]
                curr_balance = self.balance_history[i][1]
                daily_return = (curr_balance - prev_balance) / prev_balance
                daily_returns.append(daily_return)
        
        # Calculate metrics
        risk_metrics = {}
        
        if daily_returns:
            # Volatility (annualized)
            volatility = np.std(daily_returns) * np.sqrt(252)
            risk_metrics['volatility'] = volatility
            
            # Sharpe ratio
            mean_return = np.mean(daily_returns)
            sharpe_ratio = (mean_return / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0.0
            risk_metrics['sharpe_ratio'] = sharpe_ratio
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in daily_returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0.0
            sortino_ratio = (mean_return / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0.0
            risk_metrics['sortino_ratio'] = sortino_ratio
        
        # Maximum Adverse Excursion analysis
        mae_values = [t.max_adverse_excursion for t in self.swing_trades if t.max_adverse_excursion > 0]
        if mae_values:
            risk_metrics['average_mae'] = np.mean(mae_values)
            risk_metrics['max_mae'] = max(mae_values)
        
        # Maximum Favorable Excursion analysis
        mfe_values = [t.max_favorable_excursion for t in self.swing_trades if t.max_favorable_excursion > 0]
        if mfe_values:
            risk_metrics['average_mfe'] = np.mean(mfe_values)
            risk_metrics['max_mfe'] = max(mfe_values)
        
        # Risk-Reward ratios
        rr_ratios = [t.risk_reward_ratio for t in self.swing_trades if t.risk_reward_ratio > 0]
        if rr_ratios:
            risk_metrics['average_risk_reward'] = np.mean(rr_ratios)
            risk_metrics['median_risk_reward'] = np.median(rr_ratios)
        
        return risk_metrics
    
    def calculate_metrics(self) -> SwingPerformanceMetrics:
        """
        Calculate comprehensive swing trading performance metrics
        
        Returns:
            SwingPerformanceMetrics object with all performance data
        """
        try:
            if not self.swing_trades:
                logger.warning("No swing trades available for analysis")
                return self._create_empty_metrics()
            
            # Basic metrics
            total_trades = len(self.swing_trades)
            winning_trades = len([t for t in self.swing_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades) * 100
            
            # P&L metrics
            total_pnl = sum(t.pnl - t.commission for t in self.swing_trades)
            wins = [t.pnl - t.commission for t in self.swing_trades if t.pnl > 0]
            losses = [t.pnl - t.commission for t in self.swing_trades if t.pnl < 0]
            
            average_win = np.mean(wins) if wins else 0.0
            average_loss = np.mean(losses) if losses else 0.0
            largest_win = max(wins) if wins else 0.0
            largest_loss = min(losses) if losses else 0.0
            
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            # Hold time analysis
            hold_times = [(t.exit_time - t.entry_time) for t in self.swing_trades]
            average_hold_time = sum(hold_times, timedelta()) / len(hold_times) if hold_times else timedelta()
            
            # Risk metrics
            recovery_factor = total_pnl / (self.max_drawdown * self.initial_balance) if self.max_drawdown > 0 else 0.0
            
            # Calmar ratio (annual return / max drawdown)
            annual_return = (total_pnl / self.initial_balance) * (365 / max(1, len(set(t.exit_time.date() for t in self.swing_trades))))
            calmar_ratio = annual_return / self.max_drawdown if self.max_drawdown > 0 else 0.0
            
            # Detailed analysis
            swing_type_performance = self._analyze_swing_type_performance()
            regime_performance = self._analyze_regime_performance()
            hold_period_analysis = self._analyze_hold_period_performance()
            risk_metrics = self._calculate_risk_metrics()
            
            # Monthly performance
            monthly_performance = {}
            for trade in self.swing_trades:
                month_key = trade.exit_time.strftime('%Y-%m')
                net_pnl = trade.pnl - trade.commission
                if month_key not in monthly_performance:
                    monthly_performance[month_key] = 0.0
                monthly_performance[month_key] += net_pnl
            
            metrics = SwingPerformanceMetrics(
                total_trades=total_trades,
                total_pnl=total_pnl,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_hold_time=average_hold_time,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                max_drawdown=self.max_drawdown * 100,
                recovery_factor=recovery_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=risk_metrics.get('sortino_ratio', 0.0),
                swing_type_performance=swing_type_performance,
                regime_performance=regime_performance,
                hold_period_analysis=hold_period_analysis,
                monthly_performance=monthly_performance,
                risk_metrics=risk_metrics
            )
            
            logger.info(f"Swing trading metrics calculated: {total_trades} trades, "
                       f"Win rate: {win_rate:.1f}%, Total P&L: ${total_pnl:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating swing trading metrics: {str(e)}")
            raise
    
    def _create_empty_metrics(self) -> SwingPerformanceMetrics:
        """Create empty metrics for no-trade scenarios"""
        return SwingPerformanceMetrics(
            total_trades=0,
            total_pnl=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_hold_time=timedelta(),
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            max_drawdown=0.0,
            recovery_factor=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            swing_type_performance={},
            regime_performance={},
            hold_period_analysis={},
            monthly_performance={},
            risk_metrics={}
        )
    
    def get_trading_recommendations(self) -> Dict[str, Any]:
        """
        Generate trading recommendations based on swing analysis
        
        Returns:
            Dictionary with trading recommendations
        """
        metrics = self.calculate_metrics()
        recommendations = {
            "best_swing_types": [],
            "optimal_hold_periods": [],
            "preferred_market_regimes": [],
            "risk_adjustments": [],
            "strategy_improvements": []
        }
        
        # Best performing swing types
        if metrics.swing_type_performance:
            best_types = sorted(metrics.swing_type_performance.items(), 
                              key=lambda x: x[1]['profit_factor'], reverse=True)[:2]
            recommendations["best_swing_types"] = [t[0] for t in best_types]
        
        # Optimal hold periods
        if metrics.hold_period_analysis:
            best_periods = sorted(metrics.hold_period_analysis.items(), 
                                key=lambda x: x[1]['profit_factor'], reverse=True)[:2]
            recommendations["optimal_hold_periods"] = [p[0] for p in best_periods]
        
        # Preferred market regimes
        if metrics.regime_performance:
            best_regimes = sorted(metrics.regime_performance.items(), 
                                key=lambda x: x[1]['profit_factor'], reverse=True)[:2]
            recommendations["preferred_market_regimes"] = [r[0] for r in best_regimes]
        
        # Risk adjustments
        if metrics.max_drawdown > 15:
            recommendations["risk_adjustments"].append("Reduce position sizes")
        if metrics.risk_metrics.get('average_risk_reward', 0) < 1.5:
            recommendations["risk_adjustments"].append("Improve risk-reward ratios")
        
        # Strategy improvements
        if metrics.win_rate < 45:
            recommendations["strategy_improvements"].append("Focus on entry timing")
        if metrics.average_hold_time.days > 10:
            recommendations["strategy_improvements"].append("Consider shorter hold periods")
        
        return recommendations
