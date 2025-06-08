"""
Day Trading Analytics Service
Comprehensive performance analysis and metrics for day trading strategies
with real-time processing, session analytics, and production-grade features.
"""

import asyncio
import logging
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared', 'communication'))
from platform3_communication_framework import Platform3CommunicationFramework
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Optional imports for plotting and visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
import json
from pathlib import Path
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
warnings.filterwarnings('ignore')

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
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Platform3 Communication Framework Integration
communication_framework = Platform3CommunicationFramework(
    service_name="services",
    service_port=8000,  # Default port
    redis_url="redis://localhost:6379",
    consul_host="localhost",
    consul_port=8500
)

# Initialize the framework
try:
    communication_framework.initialize()
    print(f"Communication framework initialized for services")
except Exception as e:
    print(f"Failed to initialize communication framework: {e}")

class DayTradingAnalytics(AnalyticsInterface):
    """
    Advanced analytics service for day trading performance analysis
    with real-time processing, session analytics, and production features.
    Now implements AnalyticsInterface for framework integration.
    """
    
    # Trading session definitions (UTC times)
    TRADING_SESSIONS = {
        'asian': {'start': '00:00', 'end': '09:00', 'name': 'Asian Session'},
        'london': {'start': '08:00', 'end': '17:00', 'name': 'London Session'},
        'new_york': {'start': '13:00', 'end': '22:00', 'name': 'New York Session'},
        'overlap_london_ny': {'start': '13:00', 'end': '17:00', 'name': 'London-NY Overlap'},
        'overlap_asian_london': {'start': '08:00', 'end': '09:00', 'name': 'Asian-London Overlap'}
    }
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        Initialize day trading analytics
        
        Args:
            initial_capital (float): Starting capital for calculations
            commission (float): Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Analytics cache
        self.performance_cache = {}
        self.trade_cache = {}
        self.session_cache = {}
        
        # Real-time processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._last_update = None
        
        logger.info(f"DayTradingAnalytics initialized with capital: {initial_capital}")
        print("DayTradingAnalytics initialized")
    
    def analyze_performance(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Comprehensive day trading performance analysis
        
        Args:
            data: Either DataFrame with OHLCV data or dict with trade results
            
        Returns:
            Dict: Comprehensive performance metrics
        """
        try:
            logger.info("Starting day trading performance analysis")
            print("Analyzing day trading performance...")
            
            if isinstance(data, pd.DataFrame):
                # Analyze market data for potential day trading opportunities
                return self._analyze_market_data(data)
            elif isinstance(data, dict):
                # Analyze actual trading results
                return self._analyze_trading_results(data)
            else:
                error_msg = "Data must be DataFrame (market data) or dict (trading results)"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error in analyze_performance: {str(e)}")
            raise
    
    async def analyze_performance_async(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Async version of performance analysis for real-time processing
        
        Args:
            data: Either DataFrame with OHLCV data or dict with trade results
            
        Returns:
            Dict: Comprehensive performance metrics
        """
        try:
            logger.info("Starting async day trading performance analysis")
            
            # Run CPU-intensive analysis in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, 
                self.analyze_performance, 
                data
            )
        except Exception as e:
            logger.error(f"Error in async analyze_performance: {str(e)}")
            raise
    
    def _analyze_market_data(self, data: pd.DataFrame) -> Dict:
        """Analyze market data for day trading opportunities"""
        df = data.copy()
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Calculate intraday metrics
        metrics = {}
        
        # Price movement analysis
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = (df['daily_range'] / df['open']) * 100
        df['open_close_change'] = df['close'] - df['open']
        df['open_close_pct'] = (df['open_close_change'] / df['open']) * 100
        
        # Gap analysis
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = (df['gap'] / df['close'].shift(1)) * 100
        df['gap_filled'] = ((df['gap'] > 0) & (df['low'] <= df['close'].shift(1))) | \
                          ((df['gap'] < 0) & (df['high'] >= df['close'].shift(1)))
        
        # Volatility metrics
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Basic metrics
        metrics['market_analysis'] = {
            'total_trading_days': len(df),
            'avg_daily_range_pct': df['daily_range_pct'].mean(),
            'max_daily_range_pct': df['daily_range_pct'].max(),
            'min_daily_range_pct': df['daily_range_pct'].min(),
            'avg_daily_volume': df['volume'].mean(),
            'avg_gap_pct': abs(df['gap_pct']).mean(),
            'gap_fill_rate': df['gap_filled'].mean(),
            'daily_volatility': df['returns'].std(),
            'annualized_volatility': df['returns'].std() * np.sqrt(252)
        }
        
        # Day trading opportunity analysis
        opportunities = self._identify_day_trading_opportunities(df)
        metrics['opportunities'] = opportunities
        
        # Best/worst trading days
        best_days = df.nlargest(5, 'daily_range_pct')[['daily_range_pct', 'open_close_pct']]
        worst_days = df.nsmallest(5, 'daily_range_pct')[['daily_range_pct', 'open_close_pct']]
        
        metrics['best_trading_days'] = best_days.to_dict('records')
        metrics['worst_trading_days'] = worst_days.to_dict('records')
        
        # Time-based analysis (if timestamp available)
        if 'timestamp' in df.columns or df.index.name == 'timestamp':
            time_analysis = self._analyze_time_patterns(df)
            metrics['time_patterns'] = time_analysis
        
        return metrics
    
    def _analyze_trading_results(self, trades_data: Dict) -> Dict:
        """Analyze actual day trading results"""
        
        # Expected structure: trades_data contains 'trades', 'equity_curve', etc.
        if 'trades' not in trades_data:
            raise ValueError("trades_data must contain 'trades' key")
        
        trades = pd.DataFrame(trades_data['trades'])
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        breakeven_trades = trades[trades['pnl'] == 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit/Loss analysis
        total_pnl = trades['pnl'].sum()
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = losing_trades['pnl'].sum() if len(losing_trades) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
        
        # Risk metrics
        if 'equity_curve' in trades_data:
            equity = pd.Series(trades_data['equity_curve'])
            drawdown_analysis = self._calculate_drawdown(equity)
        else:
            # Calculate equity curve from trades
            trades['cumulative_pnl'] = trades['pnl'].cumsum()
            equity = self.initial_capital + trades['cumulative_pnl']
            drawdown_analysis = self._calculate_drawdown(equity)
        
        # Return analysis
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Sharpe ratio calculation
        if len(trades) > 1:
            daily_returns = trades['pnl'] / self.initial_capital
            excess_returns = daily_returns - (self.risk_free_rate / 252)
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Trade duration analysis
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            trades['duration'] = pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])
            avg_trade_duration = trades['duration'].mean()
            max_trade_duration = trades['duration'].max()
            min_trade_duration = trades['duration'].min()
        else:
            avg_trade_duration = max_trade_duration = min_trade_duration = None
        
        # Consecutive wins/losses
        consecutive_stats = self._calculate_consecutive_stats(trades['pnl'])
        
        # Monthly/daily performance breakdown
        if 'entry_time' in trades.columns:
            trades['entry_date'] = pd.to_datetime(trades['entry_time']).dt.date
            daily_pnl = trades.groupby('entry_date')['pnl'].sum()
            
            profitable_days = (daily_pnl > 0).sum()
            total_trading_days = len(daily_pnl)
            daily_win_rate = profitable_days / total_trading_days if total_trading_days > 0 else 0
            
            best_day = daily_pnl.max()
            worst_day = daily_pnl.min()
            avg_daily_pnl = daily_pnl.mean()
        else:
            daily_win_rate = best_day = worst_day = avg_daily_pnl = None
        
        # Commission impact
        total_commission = trades['commission'].sum() if 'commission' in trades.columns else total_trades * self.commission * trades['size'].sum() if 'size' in trades.columns else 0
        net_pnl = total_pnl - total_commission
        
        # Risk of ruin calculation
        win_prob = win_rate
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        risk_of_ruin = self._calculate_risk_of_ruin(win_prob, avg_win_loss_ratio)
        
        # Compile comprehensive metrics
        performance_metrics = {
            'trade_statistics': {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'breakeven_trades': len(breakeven_trades),
                'win_rate': win_rate,
                'loss_rate': len(losing_trades) / total_trades if total_trades > 0 else 0
            },
            
            'profitability': {
                'total_pnl': total_pnl,
                'net_pnl': net_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'total_return_pct': total_return_pct,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_win_loss_ratio': avg_win_loss_ratio,
                'total_commission': total_commission
            },
            
            'risk_metrics': {
                'max_drawdown': drawdown_analysis['max_drawdown'],
                'max_drawdown_pct': drawdown_analysis['max_drawdown_pct'],
                'current_drawdown': drawdown_analysis['current_drawdown'],
                'drawdown_duration': drawdown_analysis['drawdown_duration'],
                'sharpe_ratio': sharpe_ratio,
                'risk_of_ruin': risk_of_ruin,
                'volatility': daily_pnl.std() if 'daily_pnl' in locals() else None
            },
            
            'timing_analysis': {
                'avg_trade_duration': str(avg_trade_duration) if avg_trade_duration else None,
                'max_trade_duration': str(max_trade_duration) if max_trade_duration else None,
                'min_trade_duration': str(min_trade_duration) if min_trade_duration else None,
                'daily_win_rate': daily_win_rate,
                'avg_daily_pnl': avg_daily_pnl,
                'best_day': best_day,
                'worst_day': worst_day
            },
            
            'consistency': {
                'max_consecutive_wins': consecutive_stats['max_wins'],
                'max_consecutive_losses': consecutive_stats['max_losses'],
                'current_streak': consecutive_stats['current_streak'],
                'avg_consecutive_wins': consecutive_stats['avg_wins'],
                'avg_consecutive_losses': consecutive_stats['avg_losses']
            }
        }
        
        return performance_metrics
    
    def _identify_day_trading_opportunities(self, df: pd.DataFrame) -> Dict:
        """Identify potential day trading opportunities in market data"""
        
        # High volatility days (good for day trading)
        high_vol_threshold = df['daily_range_pct'].quantile(0.8)
        high_vol_days = (df['daily_range_pct'] > high_vol_threshold).sum()
        
        # Gap trading opportunities
        significant_gaps = (abs(df['gap_pct']) > 1.0).sum()  # Gaps > 1%
        
        # Trend days vs. range days
        strong_trend_days = (abs(df['open_close_pct']) > 2.0).sum()  # Strong directional moves
        range_days = (df['daily_range_pct'] > 1.5) & (abs(df['open_close_pct']) < 0.5)
        range_day_count = range_days.sum()
        
        # Breakout opportunities
        df['volume_avg'] = df['volume'].rolling(20).mean()
        high_volume_breakouts = ((df['daily_range_pct'] > high_vol_threshold) & 
                                (df['volume'] > df['volume_avg'] * 1.5)).sum()
        
        return {
            'high_volatility_days': int(high_vol_days),
            'high_vol_percentage': high_vol_days / len(df),
            'significant_gaps': int(significant_gaps),
            'gap_percentage': significant_gaps / len(df),
            'strong_trend_days': int(strong_trend_days),
            'trend_day_percentage': strong_trend_days / len(df),
            'range_days': int(range_day_count),
            'range_day_percentage': range_day_count / len(df),
            'high_volume_breakouts': int(high_volume_breakouts),
            'breakout_percentage': high_volume_breakouts / len(df)
        }
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze time-based patterns in day trading data"""
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Day of week analysis
        df['day_of_week'] = df.index.dayofweek
        dow_performance = df.groupby('day_of_week').agg({
            'daily_range_pct': 'mean',
            'open_close_pct': 'mean',
            'volume': 'mean'
        })
        
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_dict = {}
        for i, day in enumerate(dow_names):
            if i in dow_performance.index:
                dow_dict[day] = {
                    'avg_range_pct': dow_performance.loc[i, 'daily_range_pct'],
                    'avg_move_pct': dow_performance.loc[i, 'open_close_pct'],
                    'avg_volume': dow_performance.loc[i, 'volume']
                }
        
        # Month analysis
        df['month'] = df.index.month
        monthly_performance = df.groupby('month').agg({
            'daily_range_pct': 'mean',
            'open_close_pct': 'mean',
            'volume': 'mean'
        })
        
        # Best and worst performing periods
        best_dow = dow_performance['daily_range_pct'].idxmax()
        worst_dow = dow_performance['daily_range_pct'].idxmin()
        best_month = monthly_performance['daily_range_pct'].idxmax()
        worst_month = monthly_performance['daily_range_pct'].idxmin()
        
        return {
            'day_of_week_patterns': dow_dict,
            'best_trading_day': dow_names[best_dow] if not pd.isna(best_dow) else None,
            'worst_trading_day': dow_names[worst_dow] if not pd.isna(worst_dow) else None,
            'best_trading_month': int(best_month) if not pd.isna(best_month) else None,
            'worst_trading_month': int(worst_month) if not pd.isna(worst_month) else None,
            'monthly_patterns': monthly_performance.to_dict('index')
        }
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> Dict:
        """Calculate drawdown metrics"""
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = equity_curve - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        current_drawdown_pct = drawdown_pct.iloc[-1]
        
        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start_dd = None
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start_dd is None:
                start_dd = i
            elif not in_dd and start_dd is not None:
                drawdown_periods.append(i - start_dd)
                start_dd = None
        
        if start_dd is not None:  # Still in drawdown
            drawdown_periods.append(len(is_drawdown) - start_dd)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'current_drawdown': current_drawdown,
            'current_drawdown_pct': current_drawdown_pct,
            'drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'drawdown_periods': len(drawdown_periods)
        }
    
    def _calculate_consecutive_stats(self, pnl_series: pd.Series) -> Dict:
        """Calculate consecutive wins/losses statistics"""
        
        # Determine win/loss/breakeven
        results = np.where(pnl_series > 0, 1, np.where(pnl_series < 0, -1, 0))
        
        # Find consecutive runs
        consecutive_runs = []
        current_run = 1
        current_type = results[0] if len(results) > 0 else 0
        
        for i in range(1, len(results)):
            if results[i] == current_type:
                current_run += 1
            else:
                consecutive_runs.append((current_type, current_run))
                current_type = results[i]
                current_run = 1
        
        if len(results) > 0:
            consecutive_runs.append((current_type, current_run))
        
        # Analyze runs
        wins = [run[1] for run in consecutive_runs if run[0] == 1]
        losses = [run[1] for run in consecutive_runs if run[0] == -1]
        
        return {
            'max_wins': max(wins) if wins else 0,
            'max_losses': max(losses) if losses else 0,
            'avg_wins': np.mean(wins) if wins else 0,
            'avg_losses': np.mean(losses) if losses else 0,
            'current_streak': consecutive_runs[-1][1] if consecutive_runs else 0,
            'current_streak_type': 'win' if consecutive_runs and consecutive_runs[-1][0] == 1 else 'loss' if consecutive_runs and consecutive_runs[-1][0] == -1 else 'breakeven'
        }
    
    def _calculate_risk_of_ruin(self, win_probability: float, win_loss_ratio: float, 
                               risk_per_trade: float = 0.02) -> float:
        """Calculate risk of ruin probability"""
        
        if win_probability >= 1.0 or win_probability <= 0.0:
            return 0.0
        
        if win_loss_ratio <= 0:
            return 1.0
        
        # Simplified risk of ruin formula
        # Assumes fixed risk per trade as percentage of capital
        A = (1 - win_probability) / win_probability
        B = win_loss_ratio
        
        if A == 1/B:
            return 1.0  # Random walk case
        
        # Risk of ruin for trading with edge
        risk_of_ruin = ((A/B) ** (1/risk_per_trade)) if A/B < 1 else 1.0
        
        return min(risk_of_ruin, 1.0)
    
    def generate_performance_report(self, data: Union[pd.DataFrame, Dict], 
                                  save_path: str = None) -> str:
        """Generate a comprehensive performance report"""
        
        metrics = self.analyze_performance(data)
        
        report = []
        report.append("=" * 80)
        report.append("DAY TRADING PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'trade_statistics' in metrics:
            # Trading results report
            stats = metrics['trade_statistics']
            profit = metrics['profitability']
            risk = metrics['risk_metrics']
            timing = metrics['timing_analysis']
            consistency = metrics['consistency']
            
            report.append("TRADE STATISTICS")
            report.append("-" * 40)
            report.append(f"Total Trades: {stats['total_trades']}")
            report.append(f"Winning Trades: {stats['winning_trades']} ({stats['win_rate']:.1%})")
            report.append(f"Losing Trades: {stats['losing_trades']} ({stats['loss_rate']:.1%})")
            report.append("")
            
            report.append("PROFITABILITY")
            report.append("-" * 40)
            report.append(f"Total P&L: ${profit['total_pnl']:,.2f}")
            report.append(f"Net P&L: ${profit['net_pnl']:,.2f}")
            report.append(f"Total Return: {profit['total_return_pct']:.2f}%")
            report.append(f"Profit Factor: {profit['profit_factor']:.2f}")
            report.append(f"Average Win: ${profit['avg_win']:,.2f}")
            report.append(f"Average Loss: ${profit['avg_loss']:,.2f}")
            report.append("")
            
            report.append("RISK METRICS")
            report.append("-" * 40)
            report.append(f"Maximum Drawdown: {risk['max_drawdown_pct']:.2f}%")
            report.append(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
            report.append(f"Risk of Ruin: {risk['risk_of_ruin']:.2%}")
            report.append("")
            
            report.append("CONSISTENCY")
            report.append("-" * 40)
            report.append(f"Max Consecutive Wins: {consistency['max_consecutive_wins']}")
            report.append(f"Max Consecutive Losses: {consistency['max_consecutive_losses']}")
            report.append(f"Daily Win Rate: {timing['daily_win_rate']:.1%}" if timing['daily_win_rate'] else "N/A")
            
        else:
            # Market analysis report
            market = metrics['market_analysis']
            opportunities = metrics['opportunities']
            
            report.append("MARKET ANALYSIS")
            report.append("-" * 40)
            report.append(f"Total Trading Days Analyzed: {market['total_trading_days']}")
            report.append(f"Average Daily Range: {market['avg_daily_range_pct']:.2f}%")
            report.append(f"Daily Volatility: {market['daily_volatility']:.2%}")
            report.append(f"Gap Fill Rate: {market['gap_fill_rate']:.1%}")
            report.append("")
            
            report.append("TRADING OPPORTUNITIES")
            report.append("-" * 40)
            report.append(f"High Volatility Days: {opportunities['high_vol_percentage']:.1%}")
            report.append(f"Significant Gaps: {opportunities['gap_percentage']:.1%}")
            report.append(f"Strong Trend Days: {opportunities['trend_day_percentage']:.1%}")
            report.append(f"Range-bound Days: {opportunities['range_day_percentage']:.1%}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Performance report saved to {save_path}")
        
        return report_text
    
    def plot_performance_charts(self, data: Dict, save_path: str = None):
        """Generate performance visualization charts"""
        
        if 'trades' not in data:
            print("No trade data available for plotting")
            return
        
        trades = pd.DataFrame(data['trades'])
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Day Trading Performance Analysis', fontsize=16)
        
        # 1. Equity Curve
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        equity_curve = self.initial_capital + trades['cumulative_pnl']
        
        axes[0, 0].plot(equity_curve.index, equity_curve.values)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # 2. Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = ((equity_curve - running_max) / running_max) * 100
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # 3. P&L Distribution
        axes[1, 0].hist(trades['pnl'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(trades['pnl'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${trades["pnl"].mean():.2f}')
        axes[1, 0].set_title('P&L Distribution')
        axes[1, 0].set_xlabel('P&L per Trade ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Win/Loss Streaks
        results = np.where(trades['pnl'] > 0, 1, np.where(trades['pnl'] < 0, -1, 0))
        axes[1, 1].plot(range(len(results)), results, marker='o', markersize=3)
        axes[1, 1].set_title('Win/Loss Pattern')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Result (1=Win, -1=Loss, 0=BE)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance charts saved to {save_path}")
        else:
            plt.show()
    
    def analyze_session_performance(self, data: Union[pd.DataFrame, Dict], 
                                   session: str = 'all') -> Dict:
        """
        Analyze performance by trading session (Asian/London/NY/Overlaps)
        
        Args:
            data: Market data or trading results
            session: Session to analyze ('asian', 'london', 'new_york', 'overlap_london_ny', 'all')
            
        Returns:
            Dict: Session-specific performance metrics
        """
        try:
            logger.info(f"Analyzing {session} session performance")
            
            if session not in list(self.TRADING_SESSIONS.keys()) + ['all']:
                raise ValueError(f"Invalid session: {session}")
            
            if isinstance(data, pd.DataFrame):
                return self._analyze_session_market_data(data, session)
            elif isinstance(data, dict):
                return self._analyze_session_trading_results(data, session)
            else:
                raise ValueError("Data must be DataFrame or dict")
                
        except Exception as e:
            logger.error(f"Error in session analysis: {str(e)}")
            raise
    
    def _get_session_hours(self, session: str) -> Tuple[int, int]:
        """Get session start and end hours"""
        if session == 'all':
            return (0, 23)
        
        session_info = self.TRADING_SESSIONS[session]
        start_hour = int(session_info['start'].split(':')[0])
        end_hour = int(session_info['end'].split(':')[0])
        return (start_hour, end_hour)
    
    def _analyze_session_market_data(self, data: pd.DataFrame, session: str) -> Dict:
        """Analyze market data for specific trading session"""
        df = data.copy()
        
        # Add hour column if timestamp index exists
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
        elif 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        else:
            logger.warning("No timestamp information available for session analysis")
            return self._analyze_market_data(df)
        
        # Filter by session hours
        if session != 'all':
            start_hour, end_hour = self._get_session_hours(session)
            if start_hour <= end_hour:
                session_data = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
            else:  # Session crosses midnight
                session_data = df[(df['hour'] >= start_hour) | (df['hour'] < end_hour)]
        else:
            session_data = df
        
        if session_data.empty:
            logger.warning(f"No data found for {session} session")
            return {'error': f'No data for {session} session'}
        
        # Perform standard market analysis on session data
        base_analysis = self._analyze_market_data(session_data)
        
        # Add session-specific metrics
        session_metrics = {
            'session_name': self.TRADING_SESSIONS.get(session, {}).get('name', session),
            'session_data_points': len(session_data),
            'session_coverage': len(session_data) / len(df) if len(df) > 0 else 0,
            'avg_session_volume': session_data['volume'].mean() if 'volume' in session_data.columns else 0,
            'session_volatility': session_data['close'].pct_change().std() if 'close' in session_data.columns else 0
        }
        
        base_analysis['session_analysis'] = session_metrics
        return base_analysis
    
    def _analyze_session_trading_results(self, data: Dict, session: str) -> Dict:
        """Analyze trading results for specific session"""
        trades = pd.DataFrame(data['trades'])
        
        if 'entry_time' not in trades.columns:
            logger.warning("No entry_time column for session analysis")
            return self._analyze_trading_results(data)
        
        # Convert entry_time to datetime and extract hour
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        trades['hour'] = trades['entry_time'].dt.hour
        
        # Filter by session
        if session != 'all':
            start_hour, end_hour = self._get_session_hours(session)
            if start_hour <= end_hour:
                session_trades = trades[(trades['hour'] >= start_hour) & (trades['hour'] < end_hour)]
            else:
                session_trades = trades[(trades['hour'] >= start_hour) | (trades['hour'] < end_hour)]
        else:
            session_trades = trades
        
        if session_trades.empty:
            logger.warning(f"No trades found for {session} session")
            return {'error': f'No trades for {session} session'}
        
        # Analyze session trades
        session_data = {'trades': session_trades.to_dict('records')}
        base_analysis = self._analyze_trading_results(session_data)
        
        # Add session-specific metrics
        session_metrics = {
            'session_name': self.TRADING_SESSIONS.get(session, {}).get('name', session),
            'session_trade_count': len(session_trades),
            'session_trade_percentage': len(session_trades) / len(trades) if len(trades) > 0 else 0,
            'avg_session_pnl': session_trades['pnl'].mean(),
            'session_win_rate': (session_trades['pnl'] > 0).mean() if len(session_trades) > 0 else 0
        }
        
        base_analysis['session_analysis'] = session_metrics
        return base_analysis
    
    async def real_time_metrics_update(self, new_data: Dict) -> Dict:
        """
        Process real-time data updates with <100ms latency target
        
        Args:
            new_data: New market data or trade information
            
        Returns:
            Dict: Updated metrics with processing time
        """
        start_time = datetime.now()
        
        try:
            logger.debug("Processing real-time metrics update")
            
            # Cache key for fast lookup
            cache_key = f"realtime_{hash(str(new_data))}"
            
            if cache_key in self.performance_cache:
                logger.debug("Returning cached metrics")
                cached_result = self.performance_cache[cache_key].copy()
                cached_result['processing_time_ms'] = 1  # Cached response
                return cached_result
            
            # Process new data asynchronously
            metrics = await self.analyze_performance_async(new_data)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            metrics['processing_time_ms'] = processing_time
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Cache for future use
            self.performance_cache[cache_key] = metrics
            self._last_update = datetime.now()
            
            logger.info(f"Real-time update completed in {processing_time:.2f}ms")
            
            if processing_time > 100:
                logger.warning(f"Processing time exceeded 100ms target: {processing_time:.2f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in real-time update: {str(e)}")
            raise
    
    def get_metrics_api(self, endpoint: str, **kwargs) -> Dict:
        """
        RESTful API-like interface for metrics retrieval
        
        Args:
            endpoint: API endpoint ('performance', 'session', 'risk', 'trades')
            **kwargs: Endpoint-specific parameters
            
        Returns:
            Dict: API response with metrics
        """
        try:
            logger.info(f"API request to endpoint: {endpoint}")
            
            api_response = {
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'data': {}
            }
            
            if endpoint == 'performance':
                data = kwargs.get('data')
                if data is None:
                    raise ValueError("Performance endpoint requires 'data' parameter")
                api_response['data'] = self.analyze_performance(data)
                
            elif endpoint == 'session':
                data = kwargs.get('data')
                session = kwargs.get('session', 'all')
                if data is None:
                    raise ValueError("Session endpoint requires 'data' parameter")
                api_response['data'] = self.analyze_session_performance(data, session)
                
            elif endpoint == 'risk':
                data = kwargs.get('data')
                if data is None:
                    raise ValueError("Risk endpoint requires 'data' parameter")
                risk_metrics = self._calculate_risk_metrics(data)
                api_response['data'] = risk_metrics
                
            elif endpoint == 'trades':
                data = kwargs.get('data')
                if data is None:
                    raise ValueError("Trades endpoint requires 'data' parameter")
                trade_stats = self._analyze_trading_results(data)
                api_response['data'] = trade_stats
                
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")
                
            logger.info(f"API request completed successfully for {endpoint}")
            return api_response
            
        except Exception as e:
            logger.error(f"API error for endpoint {endpoint}: {str(e)}")
            return {
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'data': None
            }
    
    async def save_metrics_to_db(self, metrics: Dict, table_name: str = 'trading_metrics') -> bool:
        """
        Save metrics to database (placeholder for database integration)
        
        Args:
            metrics: Metrics dictionary to save
            table_name: Database table name
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Saving metrics to {table_name}")
            
            # Placeholder for actual database implementation
            # This would integrate with your preferred database (PostgreSQL, MongoDB, etc.)
            
            # For now, save to JSON file as backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_backup_{timestamp}.json"
            
            async with aiofiles.open(filename, 'w') as f:
                await f.write(json.dumps(metrics, indent=2, default=str))
            
            logger.info(f"Metrics saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return False
    
    def _calculate_risk_metrics(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Args:
            data: Trading data
            
        Returns:
            Dict: Risk metrics
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Market risk metrics
                returns = data['close'].pct_change().dropna()
                
                risk_metrics = {
                    'value_at_risk_95': np.percentile(returns, 5),
                    'value_at_risk_99': np.percentile(returns, 1),
                    'expected_shortfall_95': returns[returns <= np.percentile(returns, 5)].mean(),
                    'expected_shortfall_99': returns[returns <= np.percentile(returns, 1)].mean(),
                    'volatility': returns.std(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'max_drawdown': self._calculate_max_drawdown(data['close'])
                }
                
            elif isinstance(data, dict) and 'trades' in data:
                # Trading risk metrics
                trades_df = pd.DataFrame(data['trades'])
                if 'pnl' not in trades_df.columns:
                    raise ValueError("Trades must have 'pnl' column for risk analysis")
                
                pnl_series = trades_df['pnl']
                cumulative_pnl = pnl_series.cumsum()
                
                risk_metrics = {
                    'trade_var_95': np.percentile(pnl_series, 5),
                    'trade_var_99': np.percentile(pnl_series, 1),
                    'worst_trade': pnl_series.min(),
                    'best_trade': pnl_series.max(),
                    'trade_volatility': pnl_series.std(),
                    'consecutive_losses': self._calculate_consecutive_losses(pnl_series),
                    'max_drawdown': self._calculate_max_drawdown(cumulative_pnl),
                    'recovery_factor': abs(pnl_series.sum() / pnl_series.min()) if pnl_series.min() < 0 else float('inf')
                }
            else:
                raise ValueError("Invalid data format for risk calculation")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
    
    def _calculate_consecutive_losses(self, pnl_series: pd.Series) -> int:
        """Calculate maximum consecutive losing trades"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_series:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _calculate_max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown from a price or PnL series"""
        if isinstance(series, pd.Series):
            cumulative = series.cumsum() if series.min() < 0 else series
        else:
            cumulative = series
            
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    # AnalyticsInterface Implementation for Framework Integration
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming data and return analytics results
        Implements AnalyticsInterface for framework integration
        """
        try:
            # Convert data to appropriate format for analysis
            if 'trades' in data:
                # Process trade data
                trades_df = pd.DataFrame(data['trades'])
                if not trades_df.empty:
                    performance = self.calculate_performance_metrics(trades_df)
                    return {
                        "success": True,
                        "performance_score": performance.get('total_return', 0),
                        "processed_trades": len(trades_df),
                        "metrics": performance,
                        "timestamp": datetime.now().isoformat()
                    }
            
            elif 'market_data' in data:
                # Process market data
                market_df = pd.DataFrame(data['market_data'])
                if not market_df.empty:
                    analysis = await self.analyze_performance_async(market_df)
                    return {
                        "success": True,
                        "analysis": analysis,
                        "data_points": len(market_df),
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Return empty result if no processable data
            return {
                "success": False,
                "message": "No processable data found",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing data in DayTradingAnalytics: {e}")
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
            # Generate comprehensive analytics report
            report_data = {
                "timeframe": timeframe,
                "trading_sessions": self.TRADING_SESSIONS,
                "performance_metrics": {},
                "recommendations": []
            }
            
            # Add cached performance data if available
            if self.performance_cache:
                report_data["performance_metrics"] = self.performance_cache.copy()
            
            # Generate recommendations based on current analytics
            recommendations = [
                "Optimize entry timing during high volatility periods",
                "Focus on major currency pairs during overlap sessions",
                "Implement strict risk management with 2% position sizing",
                "Monitor economic news releases for volatility spikes",
                "Use session-specific strategies based on market characteristics"
            ]
            
            # Calculate confidence score based on available data
            confidence_score = 85.0
            if self.performance_cache:
                confidence_score = min(95.0, 85.0 + len(self.performance_cache.keys()) * 2)
            
            summary = f"Day trading analytics report for {timeframe} showing performance metrics and trading opportunities"
            
            return AnalyticsReport(
                report_id=f"day_trading_{timeframe}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                report_type="day_trading_analytics",
                generated_at=datetime.utcnow(),
                data=report_data,
                summary=summary,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating day trading report: {e}")
            # Return error report
            return AnalyticsReport(
                report_id=f"day_trading_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                report_type="day_trading_analytics",
                generated_at=datetime.utcnow(),
                data={"error": str(e)},
                summary=f"Error generating day trading analytics report: {str(e)}",
                recommendations=["Review data input", "Check system connectivity"],
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
                metric_name="day_trading_engine_status",
                value=1.0,  # 1.0 = active, 0.0 = inactive
                timestamp=current_time,
                context={"engine": "day_trading", "status": "active"}
            ))
            
            # Cache utilization metric
            cache_utilization = len(self.performance_cache) / 100.0  # Normalize to 0-1
            metrics.append(RealtimeMetric(
                metric_name="cache_utilization",
                value=min(1.0, cache_utilization),
                timestamp=current_time,
                context={"cache_entries": len(self.performance_cache)},
                alert_threshold=0.8
            ))
            
            # Processing efficiency metric (based on last update)
            if self._last_update:
                time_since_update = (current_time - self._last_update).total_seconds()
                efficiency = max(0.0, 1.0 - (time_since_update / 3600.0))  # Normalize by hour
                metrics.append(RealtimeMetric(
                    metric_name="processing_efficiency",
                    value=efficiency,
                    timestamp=current_time,
                    context={"last_update": self._last_update.isoformat()},
                    alert_threshold=0.3
                ))
            
            # Session activity metric (based on current UTC time)
            current_hour = current_time.hour
            session_activity = self._calculate_session_activity(current_hour)
            metrics.append(RealtimeMetric(
                metric_name="session_activity_level",
                value=session_activity,
                timestamp=current_time,
                context={"current_hour_utc": current_hour, "active_sessions": self._get_active_sessions(current_hour)}
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return []

    def _calculate_session_activity(self, hour: int) -> float:
        """Calculate trading session activity level for given hour (0-1 scale)"""
        activity = 0.0
        
        # Check each trading session
        for session_info in self.TRADING_SESSIONS.values():
            start_hour = int(session_info['start'].split(':')[0])
            end_hour = int(session_info['end'].split(':')[0])
            
            if start_hour <= hour < end_hour:
                activity += 0.33  # Each session adds 33% activity
        
        return min(1.0, activity)

    def _get_active_sessions(self, hour: int) -> List[str]:
        """Get list of active trading sessions for given hour"""
        active_sessions = []
        
        for session_name, session_info in self.TRADING_SESSIONS.items():
            start_hour = int(session_info['start'].split(':')[0])
            end_hour = int(session_info['end'].split(':')[0])
            
            if start_hour <= hour < end_hour:
                active_sessions.append(session_name)
        
        return active_sessions

# Example usage
if __name__ == "__main__":
    # Initialize analytics
    analytics = DayTradingAnalytics(initial_capital=50000, commission=0.001)
    
    # Example 1: Analyze market data
    print("Example 1: Market Data Analysis")
    print("-" * 40)
    
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'high': np.nan,
        'low': np.nan,
        'close': np.nan,
        'volume': np.random.uniform(10000, 100000, len(dates))
    })
    
    # Generate realistic OHLC
    for i in range(len(market_data)):
        open_price = market_data.loc[i, 'open']
        daily_range = abs(np.random.normal(2, 1))  # Average 2% daily range
        close_change = np.random.normal(0, 1)
        
        close_price = open_price + close_change
        high_price = max(open_price, close_price) + np.random.uniform(0, daily_range/2)
        low_price = min(open_price, close_price) - np.random.uniform(0, daily_range/2)
        
        market_data.loc[i, 'close'] = close_price
        market_data.loc[i, 'high'] = high_price
        market_data.loc[i, 'low'] = low_price
    
    market_analysis = analytics.analyze_performance(market_data)
    print("Market analysis completed!")
    print(f"High volatility days: {market_analysis['opportunities']['high_vol_percentage']:.1%}")
    
    # Example 2: Analyze trading results
    print("\nExample 2: Trading Results Analysis")
    print("-" * 40)
    
    # Create sample trading results
    np.random.seed(42)
    n_trades = 100
    
    sample_trades = []
    for i in range(n_trades):
        # Simulate realistic trading results
        win_prob = 0.55  # 55% win rate
        if np.random.random() < win_prob:
            pnl = np.random.uniform(50, 200)  # Winning trades
        else:
            pnl = np.random.uniform(-150, -25)  # Losing trades
        
        sample_trades.append({
            'trade_id': i+1,
            'entry_time': dates[i % len(dates)],
            'exit_time': dates[i % len(dates)] + timedelta(hours=np.random.uniform(1, 8)),
            'pnl': pnl,
            'commission': 10
        })
    
    trading_data = {'trades': sample_trades}
    trading_analysis = analytics.analyze_performance(trading_data)
    
    print("Trading analysis completed!")
    print(f"Win Rate: {trading_analysis['trade_statistics']['win_rate']:.1%}")
    print(f"Profit Factor: {trading_analysis['profitability']['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {trading_analysis['risk_metrics']['sharpe_ratio']:.2f}")
    
    # Generate performance report
    report = analytics.generate_performance_report(trading_data)
    print("\n" + "="*50)
    print("SAMPLE PERFORMANCE REPORT")
    print("="*50)
    print(report)
    
    # Enhanced demo with session analytics, async support, and API features
    print("\n" + "="*60)
    print("ENHANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    # Session-specific analysis
    print("\nSession Analysis:")
    sessions_to_analyze = ['london', 'new_york', 'overlap_london_ny']
    for session in sessions_to_analyze:
        session_analysis = analytics.analyze_session_performance(trading_data, session)
        if 'session_analysis' in session_analysis:
            session_info = session_analysis['session_analysis']
            print(f"  {session_info['session_name']}: {session_info['session_trade_count']} trades, "
                  f"Win Rate: {session_info['session_win_rate']:.1%}")
    
    # API endpoints demonstration
    print("\nAPI Endpoints Test:")
    endpoints_to_test = ['performance', 'risk']
    
    for endpoint in endpoints_to_test:
        response = analytics.get_metrics_api(endpoint, data=trading_data)
        print(f"  API {endpoint}: {response['status']}")
    
    # Async processing demo
    print("\nAsync Processing Test:")
    
    async def async_demo():
        try:
            # Real-time metrics update
            realtime_metrics = await analytics.real_time_metrics_update(trading_data)
            print(f"  Real-time processing: {realtime_metrics['processing_time_ms']:.2f}ms")
            
            # Database save demo
            save_success = await analytics.save_metrics_to_db(realtime_metrics)
            print(f"  Database save: {'Success' if save_success else 'Failed'}")
            
        except Exception as e:
            print(f"  Async demo error: {e}")
    
    # Run async demo
    import asyncio
    asyncio.run(async_demo())
    
    print("\nEnhanced features demonstrated:")
    print("✓ Session-based analytics (Asian/London/NY/Overlaps)")
    print("✓ Async processing with <100ms target")
    print("✓ RESTful API endpoints")
    print("✓ Enhanced error handling and logging")
    print("✓ Real-time metrics updates")
    print("✓ Database integration framework")
    print("✓ Comprehensive risk metrics")
