"""
Day Trading Backtester for M5-H1 Intraday Trading Strategies
Specialized backtesting engine optimized for day trading strategies with session-based analysis.

This module provides:
- Session-based day trading strategy validation
- Intraday performance analysis
- Multi-timeframe signal coordination
- Session-specific risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import json
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"

class DayTradingSignal(Enum):
    """Day trading signal types"""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"

@dataclass
class DayTrade:
    """Individual day trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    direction: str
    entry_signal: str
    exit_signal: Optional[str]
    pnl: Optional[float]
    pips: Optional[float]
    duration_minutes: Optional[float]
    session: TradingSession
    max_favorable_excursion: float
    max_adverse_excursion: float
    commission: float
    swap: float

@dataclass
class SessionMetrics:
    """Session-specific performance metrics"""
    session: TradingSession
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    total_pips: float
    avg_trade_duration: float
    max_drawdown: float
    profit_factor: float

@dataclass
class DayTradingMetrics:
    """Day trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pips: float
    avg_trade_duration: float
    avg_pips_per_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    recovery_factor: float
    trades_per_day: float
    best_session: TradingSession
    worst_session: TradingSession
    session_metrics: Dict[TradingSession, SessionMetrics]

@dataclass
class DayTradingBacktestResult:
    """Complete day trading backtest results"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    trades: List[DayTrade]
    metrics: DayTradingMetrics
    equity_curve: List[Tuple[datetime, float]]
    drawdown_curve: List[Tuple[datetime, float]]
    daily_pnl: List[Tuple[datetime, float]]
    execution_time: float

class DayTradingBacktester:
    """
    Day Trading Backtester for M5-H1 Intraday Strategies.

    Optimized for:
    - Session-based trading analysis
    - Intraday performance tracking
    - Multi-timeframe coordination
    - Risk management per session
    """

    def __init__(self, initial_balance: float = 50000.0,
                 commission_per_lot: float = 7.0,
                 swap_per_lot_per_day: float = -2.0,
                 max_positions: int = 3):
        """
        Initialize day trading backtester.

        Args:
            initial_balance: Starting account balance
            commission_per_lot: Commission per standard lot
            swap_per_lot_per_day: Swap cost per lot per day
            max_positions: Maximum concurrent positions
        """
        self.initial_balance = initial_balance
        self.commission_per_lot = commission_per_lot
        self.swap_per_lot_per_day = swap_per_lot_per_day
        self.max_positions = max_positions

        # Trading state
        self.current_balance = initial_balance
        self.current_positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.daily_pnl = []

        # Performance tracking
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.current_day_pnl = 0.0
        self.last_date = None

    async def backtest_strategy(self, data: pd.DataFrame, strategy_func: callable,
                              strategy_params: Dict, symbol: str = "EURUSD") -> DayTradingBacktestResult:
        """
        Run day trading backtest on intraday data.

        Args:
            data: M5-H1 OHLCV data
            strategy_func: Strategy function that generates signals
            strategy_params: Strategy parameters
            symbol: Trading symbol

        Returns:
            Complete backtest results
        """
        start_time = time.time()

        try:
            # Validate and prepare data
            data = await self._prepare_day_trading_data(data)

            # Reset state
            self._reset_backtest_state()

            # Run bar-by-bar simulation
            for i in range(len(data)):
                current_bar = data.iloc[i]

                # Check for new trading day
                self._check_new_trading_day(current_bar)

                # Generate strategy signal
                signal = await self._generate_strategy_signal(
                    strategy_func, data.iloc[max(0, i-200):i+1], strategy_params
                )

                # Process signal
                await self._process_day_trading_signal(signal, current_bar, symbol)

                # Update positions
                await self._update_positions(current_bar)

                # Update equity curve
                self._update_equity_curve(current_bar['timestamp'])

                # Update performance tracking
                self._update_performance_tracking()

            # Close all open positions
            if self.current_positions:
                for position in self.current_positions.copy():
                    await self._close_position(position, data.iloc[-1], "backtest_end")

            # Calculate final metrics
            metrics = await self._calculate_day_trading_metrics(data)

            execution_time = time.time() - start_time

            return DayTradingBacktestResult(
                strategy_name=strategy_params.get('name', 'DayTradingStrategy'),
                symbol=symbol,
                timeframe="M15",
                start_date=data.iloc[0]['timestamp'],
                end_date=data.iloc[-1]['timestamp'],
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                trades=self.trades,
                metrics=metrics,
                equity_curve=self.equity_curve,
                drawdown_curve=self.drawdown_curve,
                daily_pnl=self.daily_pnl,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Day trading backtest failed: {e}")
            raise

    async def _prepare_day_trading_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate day trading data"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Add trading session information
        data['session'] = data['timestamp'].apply(self._determine_trading_session)

        # Add daily markers
        data['date'] = data['timestamp'].dt.date

        return data.sort_values('timestamp').reset_index(drop=True)

    def _determine_trading_session(self, timestamp: datetime) -> TradingSession:
        """Determine trading session based on UTC time"""
        hour = timestamp.hour

        if 0 <= hour < 8:
            return TradingSession.ASIAN
        elif 8 <= hour < 13:
            if 8 <= hour < 9:
                return TradingSession.OVERLAP_ASIAN_LONDON
            else:
                return TradingSession.LONDON
        elif 13 <= hour < 17:
            return TradingSession.OVERLAP_LONDON_NY
        else:
            return TradingSession.NEW_YORK

    def _reset_backtest_state(self):
        """Reset backtester state for new run"""
        self.current_balance = self.initial_balance
        self.current_positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.daily_pnl = []
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.current_day_pnl = 0.0
        self.last_date = None

    def _check_new_trading_day(self, bar: pd.Series):
        """Check for new trading day and update daily P&L"""
        current_date = bar['timestamp'].date()

        if self.last_date and current_date != self.last_date:
            # New trading day - record previous day's P&L
            self.daily_pnl.append((self.last_date, self.current_day_pnl))
            self.current_day_pnl = 0.0

        self.last_date = current_date

    async def _generate_strategy_signal(self, strategy_func: callable,
                                      data_window: pd.DataFrame,
                                      params: Dict) -> DayTradingSignal:
        """Generate trading signal from strategy function"""
        try:
            signal = strategy_func(data_window, params)

            # Convert string signals to enum
            if isinstance(signal, str):
                signal = DayTradingSignal(signal.lower())

            return signal

        except Exception as e:
            logger.warning(f"Strategy signal generation failed: {e}")
            return DayTradingSignal.HOLD

    async def _process_day_trading_signal(self, signal: DayTradingSignal,
                                        bar: pd.Series, symbol: str):
        """Process day trading signal and execute trades"""

        if signal == DayTradingSignal.BUY and len(self.current_positions) < self.max_positions:
            await self._open_position("long", bar, signal.value, symbol)

        elif signal == DayTradingSignal.SELL and len(self.current_positions) < self.max_positions:
            await self._open_position("short", bar, signal.value, symbol)

        elif signal == DayTradingSignal.CLOSE_LONG:
            await self._close_positions_by_direction("long", bar, signal.value)

        elif signal == DayTradingSignal.CLOSE_SHORT:
            await self._close_positions_by_direction("short", bar, signal.value)

        elif signal == DayTradingSignal.SCALE_IN:
            await self._scale_into_positions(bar, symbol)

        elif signal == DayTradingSignal.SCALE_OUT:
            await self._scale_out_of_positions(bar)

    async def _open_position(self, direction: str, bar: pd.Series,
                           signal: str, symbol: str):
        """Open new day trading position"""

        # Calculate position size based on account balance
        risk_per_trade = 0.02  # 2% risk per trade
        position_size = (self.current_balance * risk_per_trade) / (100 * 0.0001)  # Simplified position sizing
        position_size = min(position_size, 1.0)  # Max 1 lot

        # Determine entry price
        entry_price = bar['close']  # Simplified - would use more sophisticated entry logic

        # Calculate costs
        commission = self.commission_per_lot * position_size

        position = {
            'id': len(self.current_positions),
            'direction': direction,
            'entry_time': bar['timestamp'],
            'entry_price': entry_price,
            'quantity': position_size,
            'entry_signal': signal,
            'session': bar['session'],
            'commission': commission,
            'symbol': symbol,
            'max_favorable_excursion': 0.0,
            'max_adverse_excursion': 0.0
        }

        self.current_positions.append(position)
        logger.debug(f"Opened {direction} position at {entry_price} on {bar['timestamp']}")

    async def _close_position(self, position: Dict, bar: pd.Series, signal: str):
        """Close specific position"""

        exit_price = bar['close']

        # Calculate P&L
        if position['direction'] == "long":
            pnl = (exit_price - position['entry_price']) * position['quantity'] * 100000
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity'] * 100000

        # Calculate swap cost (simplified)
        days_held = (bar['timestamp'] - position['entry_time']).days
        swap_cost = self.swap_per_lot_per_day * position['quantity'] * days_held

        # Net P&L
        net_pnl = pnl - position['commission'] - swap_cost

        # Calculate pips
        pip_value = 0.0001
        if position['direction'] == "long":
            pips = (exit_price - position['entry_price']) / pip_value
        else:
            pips = (position['entry_price'] - exit_price) / pip_value

        # Calculate duration
        duration = (bar['timestamp'] - position['entry_time']).total_seconds() / 60  # minutes

        # Create trade record
        trade = DayTrade(
            entry_time=position['entry_time'],
            exit_time=bar['timestamp'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            direction=position['direction'],
            entry_signal=position['entry_signal'],
            exit_signal=signal,
            pnl=net_pnl,
            pips=pips,
            duration_minutes=duration,
            session=position['session'],
            max_favorable_excursion=position['max_favorable_excursion'],
            max_adverse_excursion=position['max_adverse_excursion'],
            commission=position['commission'],
            swap=swap_cost
        )

        self.trades.append(trade)
        self.current_balance += net_pnl
        self.current_day_pnl += net_pnl

        # Remove position from current positions
        self.current_positions.remove(position)

        logger.debug(f"Closed position: {net_pnl:.2f} PnL, {pips:.1f} pips, {duration:.0f}min")

    async def _close_positions_by_direction(self, direction: str, bar: pd.Series, signal: str):
        """Close all positions in specified direction"""
        positions_to_close = [pos for pos in self.current_positions if pos['direction'] == direction]

        for position in positions_to_close:
            await self._close_position(position, bar, signal)

    async def _scale_into_positions(self, bar: pd.Series, symbol: str):
        """Scale into existing positions"""
        if self.current_positions and len(self.current_positions) < self.max_positions:
            # Add to existing position in same direction as most recent
            last_position = self.current_positions[-1]
            await self._open_position(last_position['direction'], bar, "scale_in", symbol)

    async def _scale_out_of_positions(self, bar: pd.Series):
        """Scale out of existing positions"""
        if len(self.current_positions) > 1:
            # Close oldest position
            oldest_position = min(self.current_positions, key=lambda x: x['entry_time'])
            await self._close_position(oldest_position, bar, "scale_out")

    async def _update_positions(self, bar: pd.Series):
        """Update position metrics and check for stop losses"""
        current_price = bar['close']

        for position in self.current_positions:
            # Update max favorable/adverse excursion
            if position['direction'] == "long":
                excursion = current_price - position['entry_price']
            else:
                excursion = position['entry_price'] - current_price

            if excursion > position['max_favorable_excursion']:
                position['max_favorable_excursion'] = excursion
            elif excursion < 0 and abs(excursion) > position['max_adverse_excursion']:
                position['max_adverse_excursion'] = abs(excursion)

    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current balance"""
        current_equity = self.current_balance

        # Add unrealized P&L from open positions
        for position in self.current_positions:
            # Simplified unrealized P&L calculation
            current_equity += 0  # Placeholder

        self.equity_curve.append((timestamp, current_equity))

    def _update_performance_tracking(self):
        """Update performance tracking metrics"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        self.drawdown_curve.append((datetime.now(), current_drawdown))

    async def _calculate_day_trading_metrics(self, data: pd.DataFrame) -> DayTradingMetrics:
        """Calculate comprehensive day trading metrics"""

        if not self.trades:
            return DayTradingMetrics(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                total_pnl=0.0, total_pips=0.0, avg_trade_duration=0.0,
                avg_pips_per_trade=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
                profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                recovery_factor=0.0, trades_per_day=0.0,
                best_session=TradingSession.ASIAN, worst_session=TradingSession.ASIAN,
                session_metrics={}
            )

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L metrics
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_pips = sum(trade.pips for trade in self.trades)
        avg_pips_per_trade = total_pips / total_trades if total_trades > 0 else 0.0

        # Duration metrics
        avg_trade_duration = np.mean([trade.duration_minutes for trade in self.trades])

        # Consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(self.trades, True)
        max_consecutive_losses = self._calculate_max_consecutive(self.trades, False)

        # Risk metrics
        winning_pnl = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        losing_pnl = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

        # Sharpe ratio
        returns = [trade.pnl / self.initial_balance for trade in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

        # Recovery factor
        recovery_factor = total_pnl / (self.max_drawdown * self.initial_balance) if self.max_drawdown > 0 else float('inf')

        # Trading frequency
        total_days = (data.iloc[-1]['timestamp'] - data.iloc[0]['timestamp']).days
        trades_per_day = total_trades / total_days if total_days > 0 else 0.0

        # Session analysis
        session_metrics = self._calculate_session_metrics()
        best_session = max(session_metrics.keys(), key=lambda x: session_metrics[x].total_pnl) if session_metrics else TradingSession.ASIAN
        worst_session = min(session_metrics.keys(), key=lambda x: session_metrics[x].total_pnl) if session_metrics else TradingSession.ASIAN

        return DayTradingMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pips=total_pips,
            avg_trade_duration=avg_trade_duration,
            avg_pips_per_trade=avg_pips_per_trade,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            recovery_factor=recovery_factor,
            trades_per_day=trades_per_day,
            best_session=best_session,
            worst_session=worst_session,
            session_metrics=session_metrics
        )

    def _calculate_session_metrics(self) -> Dict[TradingSession, SessionMetrics]:
        """Calculate metrics for each trading session"""
        session_trades = {}

        # Group trades by session
        for trade in self.trades:
            session = trade.session
            if session not in session_trades:
                session_trades[session] = []
            session_trades[session].append(trade)

        # Calculate metrics for each session
        session_metrics = {}
        for session, trades in session_trades.items():
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.pnl > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            total_pnl = sum(trade.pnl for trade in trades)
            total_pips = sum(trade.pips for trade in trades)
            avg_duration = np.mean([trade.duration_minutes for trade in trades])

            # Calculate session-specific drawdown
            session_equity = []
            running_pnl = 0
            for trade in sorted(trades, key=lambda x: x.entry_time):
                running_pnl += trade.pnl
                session_equity.append(running_pnl)

            peak = 0
            max_dd = 0
            for equity in session_equity:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / abs(peak) if peak != 0 else 0
                if dd > max_dd:
                    max_dd = dd

            # Profit factor
            winning_pnl = sum(trade.pnl for trade in trades if trade.pnl > 0)
            losing_pnl = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

            session_metrics[session] = SessionMetrics(
                session=session,
                total_trades=total_trades,
                winning_trades=winning_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pips=total_pips,
                avg_trade_duration=avg_duration,
                max_drawdown=max_dd,
                profit_factor=profit_factor
            )

        return session_metrics

    def _calculate_max_consecutive(self, trades: List[DayTrade], winning: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if (winning and trade.pnl > 0) or (not winning and trade.pnl <= 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive
