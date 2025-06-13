"""
Day Trading Backtesting Engine
M15-H1 session-based backtesting with advanced execution modeling.

This module provides comprehensive backtesting for day trading strategies
with session-aware logic and realistic market conditions simulation.
"""

import asyncio
import numpy as np
import pandas as pd

import sys
import os
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import pytz


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

class TradingSession(Enum):
    """Trading session definitions"""
    ASIAN = "Asian"
    LONDON = "London"
    NEW_YORK = "New_York"
    OVERLAP_LONDON_NY = "London_NY_Overlap"

@dataclass
class SessionData:
    """Market data for a specific trading session"""
    session: TradingSession
    start_time: datetime
    end_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    volatility: float

@dataclass
class DayTradingResult:
    """Results from day trading backtesting"""
    strategy_name: str
    symbol: str
    timeframe: str
    session_performance: Dict[TradingSession, Dict[str, float]]
    total_trades: int
    session_trades: Dict[TradingSession, int]
    win_rate_by_session: Dict[TradingSession, float]
    total_pnl: float
    max_intraday_drawdown: float
    average_trade_duration_minutes: float
    best_session: TradingSession
    execution_time_ms: float

class DayTradingBacktester:
    """
    Advanced backtesting engine for M15-H1 day trading strategies.
    
    Features:
    - Session-based analysis (Asian, London, NY, Overlaps)
    - Intraday momentum and breakout detection
    - Volatility-adjusted position sizing
    - Session-specific performance metrics
    - Advanced risk management simulation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Session definitions (UTC times)
        self.session_times = {
            TradingSession.ASIAN: (0, 9),      # 00:00 - 09:00 UTC
            TradingSession.LONDON: (8, 17),    # 08:00 - 17:00 UTC  
            TradingSession.NEW_YORK: (13, 22), # 13:00 - 22:00 UTC
            TradingSession.OVERLAP_LONDON_NY: (13, 17)  # 13:00 - 17:00 UTC
        }
        
        # Day trading specific parameters
        self.max_trade_duration_hours = config.get('max_trade_duration_hours', 8)
        self.session_close_exit = config.get('session_close_exit', True)
        self.volatility_adjustment = config.get('volatility_adjustment', True)
        
        # Risk management
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% max daily loss
        self.max_positions_per_session = config.get('max_positions_per_session', 3)
        
    async def backtest_day_trading_strategy(
        self,
        strategy_func: callable,
        market_data: List[SessionData],
        initial_balance: float = 10000.0,
        base_position_size: float = 0.01,
        strategy_params: Dict[str, Any] = None
    ) -> DayTradingResult:
        """
        Execute comprehensive day trading backtesting with session analysis.
        
        Args:
            strategy_func: Day trading strategy function
            market_data: List of session-based market data
            initial_balance: Starting account balance
            base_position_size: Base position size (adjusted by volatility)
            strategy_params: Strategy-specific parameters
            
        Returns:
            DayTradingResult: Comprehensive day trading results
        """
        start_time = datetime.now()
        
        # Initialize tracking variables
        balance = initial_balance
        daily_balance = initial_balance
        trades = []
        session_stats = {session: {'trades': [], 'pnl': 0.0} for session in TradingSession}
        active_positions = []
        
        # Group data by trading days
        daily_data = self._group_by_trading_days(market_data)
        
        for trading_day, day_sessions in daily_data.items():
            daily_start_balance = balance
            daily_trades = []
            
            # Process each session in the trading day
            for session_data in day_sessions:
                session_trades = await self._process_trading_session(
                    strategy_func,
                    session_data,
                    balance,
                    base_position_size,
                    strategy_params,
                    active_positions
                )
                
                # Update session statistics
                session_stats[session_data.session]['trades'].extend(session_trades)
                session_pnl = sum(trade['pnl'] for trade in session_trades)
                session_stats[session_data.session]['pnl'] += session_pnl
                
                daily_trades.extend(session_trades)
                balance += session_pnl
                
                # Check daily loss limit
                daily_loss = (balance - daily_start_balance) / daily_start_balance
                if daily_loss <= -self.max_daily_loss:
                    self.logger.warning(f"Daily loss limit reached: {daily_loss:.2%}")
                    break
            
            # Close any remaining positions at end of day
            if self.session_close_exit:
                eod_trades = self._close_end_of_day_positions(active_positions, day_sessions[-1])
                daily_trades.extend(eod_trades)
                balance += sum(trade['pnl'] for trade in eod_trades)
                active_positions.clear()
            
            trades.extend(daily_trades)
        
        # Calculate performance metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Analyze session performance
        session_performance = self._analyze_session_performance(session_stats)
        
        return DayTradingResult(
            strategy_name=strategy_params.get('name', 'DayTrading') if strategy_params else 'DayTrading',
            symbol=market_data[0].session.value if market_data else 'UNKNOWN',
            timeframe='M15-H1',
            session_performance=session_performance,
            total_trades=len(trades),
            session_trades={session: len(stats['trades']) for session, stats in session_stats.items()},
            win_rate_by_session=self._calculate_session_win_rates(session_stats),
            total_pnl=sum(trade['pnl'] for trade in trades),
            max_intraday_drawdown=self._calculate_max_intraday_drawdown(trades),
            average_trade_duration_minutes=self._calculate_average_duration(trades),
            best_session=self._find_best_session(session_performance),
            execution_time_ms=execution_time
        )
    
    def _group_by_trading_days(self, market_data: List[SessionData]) -> Dict[str, List[SessionData]]:
        """Group market data by trading days"""
        daily_data = {}
        
        for session_data in market_data:
            day_key = session_data.start_time.strftime('%Y-%m-%d')
            if day_key not in daily_data:
                daily_data[day_key] = []
            daily_data[day_key].append(session_data)
        
        # Sort sessions within each day
        for day_key in daily_data:
            daily_data[day_key].sort(key=lambda x: x.start_time)
        
        return daily_data
    
    async def _process_trading_session(
        self,
        strategy_func: callable,
        session_data: SessionData,
        current_balance: float,
        base_position_size: float,
        strategy_params: Dict[str, Any],
        active_positions: List[Dict]
    ) -> List[Dict]:
        """Process a single trading session"""
        
        session_trades = []
        
        # Generate trading signals for the session
        signals = strategy_func(session_data, strategy_params)
        
        # Adjust position size based on volatility
        adjusted_position_size = self._adjust_position_size(
            base_position_size, session_data.volatility
        ) if self.volatility_adjustment else base_position_size
        
        # Process signals
        for signal in signals:
            if signal['action'] == 'BUY' or signal['action'] == 'SELL':
                # Check position limits
                if len(active_positions) >= self.max_positions_per_session:
                    continue
                
                # Execute entry
                trade = self._execute_entry(
                    signal, session_data, adjusted_position_size, current_balance
                )
                if trade:
                    active_positions.append(trade)
                    
            elif signal['action'] == 'CLOSE':
                # Close matching positions
                closed_trades = self._close_positions(
                    active_positions, signal, session_data
                )
                session_trades.extend(closed_trades)
        
        # Check for session-end exits
        if self.session_close_exit:
            remaining_trades = self._close_session_positions(active_positions, session_data)
            session_trades.extend(remaining_trades)
        
        return session_trades
    
    def _adjust_position_size(self, base_size: float, volatility: float) -> float:
        """Adjust position size based on market volatility"""
        # Reduce position size in high volatility environments
        volatility_factor = 1.0 / (1.0 + volatility * 2.0)
        return base_size * volatility_factor
    
    def _execute_entry(
        self,
        signal: Dict,
        session_data: SessionData,
        position_size: float,
        balance: float
    ) -> Optional[Dict]:
        """Execute trade entry with realistic execution modeling"""
        
        # Simulate execution price with slippage
        execution_price = self._simulate_session_execution(
            signal['price'], signal['action'], session_data
        )
        
        # Create position
        position = {
            'entry_time': signal['timestamp'],
            'entry_price': execution_price,
            'direction': 1 if signal['action'] == 'BUY' else -1,
            'size': position_size,
            'session': session_data.session,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit')
        }
        
        return position
    
    def _simulate_session_execution(
        self, price: float, action: str, session_data: SessionData
    ) -> float:
        """Simulate execution with session-specific characteristics"""
        
        # Session-based slippage factors
        slippage_factors = {
            TradingSession.ASIAN: 0.5,      # Lower liquidity
            TradingSession.LONDON: 0.3,     # High liquidity
            TradingSession.NEW_YORK: 0.3,   # High liquidity
            TradingSession.OVERLAP_LONDON_NY: 0.2  # Highest liquidity
        }
        
        base_slippage = slippage_factors.get(session_data.session, 0.4)
        volatility_slippage = session_data.volatility * 0.1
        
        total_slippage = (base_slippage + volatility_slippage) * np.random.normal(1.0, 0.3)
        total_slippage = max(0.1, total_slippage)  # Minimum slippage
        
        if action == 'BUY':
            return price + total_slippage
        else:
            return price - total_slippage
    
    def _close_positions(
        self, active_positions: List[Dict], signal: Dict, session_data: SessionData
    ) -> List[Dict]:
        """Close positions based on exit signals"""
        
        closed_trades = []
        positions_to_remove = []
        
        for i, position in enumerate(active_positions):
            # Check if this position should be closed
            if self._should_close_position(position, signal, session_data):
                
                exit_price = self._simulate_session_execution(
                    signal['price'], 
                    'SELL' if position['direction'] > 0 else 'BUY',
                    session_data
                )
                
                pnl = self._calculate_day_trading_pnl(position, exit_price)
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': signal['timestamp'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'direction': position['direction'],
                    'size': position['size'],
                    'pnl': pnl,
                    'session': position['session'],
                    'duration_minutes': (signal['timestamp'] - position['entry_time']).total_seconds() / 60
                }
                
                closed_trades.append(trade)
                positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            active_positions.pop(i)
        
        return closed_trades
    
    def _should_close_position(
        self, position: Dict, signal: Dict, session_data: SessionData
    ) -> bool:
        """Determine if a position should be closed"""
        
        # Check stop loss and take profit
        current_price = signal['price']
        
        if position['direction'] > 0:  # Long position
            if position['stop_loss'] and current_price <= position['stop_loss']:
                return True
            if position['take_profit'] and current_price >= position['take_profit']:
                return True
        else:  # Short position
            if position['stop_loss'] and current_price >= position['stop_loss']:
                return True
            if position['take_profit'] and current_price <= position['take_profit']:
                return True
        
        # Check maximum trade duration
        duration_hours = (signal['timestamp'] - position['entry_time']).total_seconds() / 3600
        if duration_hours >= self.max_trade_duration_hours:
            return True
        
        return False
    
    def _close_session_positions(
        self, active_positions: List[Dict], session_data: SessionData
    ) -> List[Dict]:
        """Close all positions at session end"""
        
        closed_trades = []
        
        for position in active_positions[:]:  # Copy list to avoid modification during iteration
            exit_price = session_data.close_price
            pnl = self._calculate_day_trading_pnl(position, exit_price)
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': session_data.end_time,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'direction': position['direction'],
                'size': position['size'],
                'pnl': pnl,
                'session': position['session'],
                'duration_minutes': (session_data.end_time - position['entry_time']).total_seconds() / 60,
                'exit_reason': 'session_close'
            }
            
            closed_trades.append(trade)
        
        active_positions.clear()
        return closed_trades
    
    def _close_end_of_day_positions(
        self, active_positions: List[Dict], last_session: SessionData
    ) -> List[Dict]:
        """Close all remaining positions at end of trading day"""
        return self._close_session_positions(active_positions, last_session)
    
    def _calculate_day_trading_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L for day trading position"""
        price_diff = exit_price - position['entry_price']
        if position['direction'] < 0:  # Short position
            price_diff = -price_diff
        return price_diff * position['size'] * 100000  # Standard lot calculation
    
    def _analyze_session_performance(
        self, session_stats: Dict[TradingSession, Dict]
    ) -> Dict[TradingSession, Dict[str, float]]:
        """Analyze performance by trading session"""
        
        performance = {}
        
        for session, stats in session_stats.items():
            trades = stats['trades']
            
            if not trades:
                performance[session] = {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl_per_trade': 0.0,
                    'avg_duration_minutes': 0.0
                }
                continue
            
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            total_pnl = sum(t['pnl'] for t in trades)
            avg_duration = sum(t['duration_minutes'] for t in trades) / len(trades)
            
            performance[session] = {
                'total_trades': len(trades),
                'win_rate': winning_trades / len(trades),
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / len(trades),
                'avg_duration_minutes': avg_duration
            }
        
        return performance
    
    def _calculate_session_win_rates(
        self, session_stats: Dict[TradingSession, Dict]
    ) -> Dict[TradingSession, float]:
        """Calculate win rates by session"""
        
        win_rates = {}
        
        for session, stats in session_stats.items():
            trades = stats['trades']
            if trades:
                winning_trades = len([t for t in trades if t['pnl'] > 0])
                win_rates[session] = winning_trades / len(trades)
            else:
                win_rates[session] = 0.0
        
        return win_rates
    
    def _calculate_max_intraday_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum intraday drawdown"""
        if not trades:
            return 0.0
        
        # Sort trades by time
        sorted_trades = sorted(trades, key=lambda x: x['entry_time'])
        
        cumulative_pnl = 0.0
        max_pnl = 0.0
        max_drawdown = 0.0
        
        for trade in sorted_trades:
            cumulative_pnl += trade['pnl']
            max_pnl = max(max_pnl, cumulative_pnl)
            drawdown = max_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_average_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in minutes"""
        if not trades:
            return 0.0
        
        total_duration = sum(trade['duration_minutes'] for trade in trades)
        return total_duration / len(trades)
    
    def _find_best_session(
        self, session_performance: Dict[TradingSession, Dict[str, float]]
    ) -> TradingSession:
        """Find the best performing trading session"""
        
        best_session = TradingSession.LONDON  # Default
        best_pnl = float('-inf')
        
        for session, performance in session_performance.items():
            if performance['total_pnl'] > best_pnl:
                best_pnl = performance['total_pnl']
                best_session = session
        
        return best_session
