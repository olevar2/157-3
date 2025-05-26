"""
Swing Trading Backtester for H1-D1 Multi-Day Trading Strategies
Specialized backtesting engine optimized for swing trading strategies with pattern-based analysis.

This module provides:
- Multi-day swing trading strategy validation
- Pattern-based entry and exit analysis
- Risk management over extended holding periods
- Weekly and monthly performance analysis
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

class SwingPattern(Enum):
    """Swing trading pattern types"""
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    RANGE_BOUND = "range_bound"

class SwingSignal(Enum):
    """Swing trading signal types"""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"
    ADD_TO_POSITION = "add_to_position"
    REDUCE_POSITION = "reduce_position"

@dataclass
class SwingTrade:
    """Individual swing trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    direction: str
    entry_signal: str
    exit_signal: Optional[str]
    pattern: SwingPattern
    pnl: Optional[float]
    pips: Optional[float]
    duration_days: Optional[float]
    max_favorable_excursion: float
    max_adverse_excursion: float
    commission: float
    swap_total: float
    risk_reward_ratio: Optional[float]

@dataclass
class WeeklyMetrics:
    """Weekly performance metrics"""
    week_start: datetime
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    total_pips: float
    max_drawdown: float
    profit_factor: float

@dataclass
class SwingTradingMetrics:
    """Swing trading performance metrics"""
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
    trades_per_week: float
    avg_risk_reward_ratio: float
    best_pattern: SwingPattern
    worst_pattern: SwingPattern
    pattern_metrics: Dict[SwingPattern, Dict]
    weekly_metrics: List[WeeklyMetrics]

@dataclass
class SwingBacktestResult:
    """Complete swing trading backtest results"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    trades: List[SwingTrade]
    metrics: SwingTradingMetrics
    equity_curve: List[Tuple[datetime, float]]
    drawdown_curve: List[Tuple[datetime, float]]
    weekly_pnl: List[Tuple[datetime, float]]
    execution_time: float

class SwingBacktester:
    """
    Swing Trading Backtester for H1-D1 Multi-Day Strategies.

    Optimized for:
    - Multi-day position holding
    - Pattern-based trading analysis
    - Extended risk management
    - Weekly/monthly performance tracking
    """

    def __init__(self, initial_balance: float = 100000.0,
                 commission_per_lot: float = 7.0,
                 swap_per_lot_per_day: float = -2.0,
                 max_positions: int = 5,
                 max_risk_per_trade: float = 0.02):
        """
        Initialize swing trading backtester.

        Args:
            initial_balance: Starting account balance
            commission_per_lot: Commission per standard lot
            swap_per_lot_per_day: Swap cost per lot per day
            max_positions: Maximum concurrent positions
            max_risk_per_trade: Maximum risk per trade (2%)
        """
        self.initial_balance = initial_balance
        self.commission_per_lot = commission_per_lot
        self.swap_per_lot_per_day = swap_per_lot_per_day
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade

        # Trading state
        self.current_balance = initial_balance
        self.current_positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.weekly_pnl = []

        # Performance tracking
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.current_week_pnl = 0.0
        self.last_week = None

    async def backtest_strategy(self, data: pd.DataFrame, strategy_func: callable,
                              strategy_params: Dict, symbol: str = "EURUSD") -> SwingBacktestResult:
        """
        Run swing trading backtest on H1-D1 data.

        Args:
            data: H1-D1 OHLCV data
            strategy_func: Strategy function that generates signals
            strategy_params: Strategy parameters
            symbol: Trading symbol

        Returns:
            Complete backtest results
        """
        start_time = time.time()

        try:
            # Validate and prepare data
            data = await self._prepare_swing_data(data)

            # Reset state
            self._reset_backtest_state()

            # Run bar-by-bar simulation
            for i in range(len(data)):
                current_bar = data.iloc[i]

                # Check for new trading week
                self._check_new_trading_week(current_bar)

                # Generate strategy signal
                signal, pattern = await self._generate_strategy_signal(
                    strategy_func, data.iloc[max(0, i-500):i+1], strategy_params
                )

                # Process signal
                await self._process_swing_signal(signal, pattern, current_bar, symbol)

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
            metrics = await self._calculate_swing_metrics(data)

            execution_time = time.time() - start_time

            return SwingBacktestResult(
                strategy_name=strategy_params.get('name', 'SwingTradingStrategy'),
                symbol=symbol,
                timeframe="H4",
                start_date=data.iloc[0]['timestamp'],
                end_date=data.iloc[-1]['timestamp'],
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                trades=self.trades,
                metrics=metrics,
                equity_curve=self.equity_curve,
                drawdown_curve=self.drawdown_curve,
                weekly_pnl=self.weekly_pnl,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Swing trading backtest failed: {e}")
            raise

    async def _prepare_swing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate swing trading data"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Add weekly markers
        data['week'] = data['timestamp'].dt.isocalendar().week
        data['year'] = data['timestamp'].dt.year

        # Calculate additional swing trading indicators
        data['atr'] = self._calculate_atr(data, 14)
        data['support'] = data['low'].rolling(20).min()
        data['resistance'] = data['high'].rolling(20).max()

        return data.sort_values('timestamp').reset_index(drop=True)

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()

    def _reset_backtest_state(self):
        """Reset backtester state for new run"""
        self.current_balance = self.initial_balance
        self.current_positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.weekly_pnl = []
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.current_week_pnl = 0.0
        self.last_week = None

    def _check_new_trading_week(self, bar: pd.Series):
        """Check for new trading week and update weekly P&L"""
        current_week = (bar['year'], bar['week'])

        if self.last_week and current_week != self.last_week:
            # New trading week - record previous week's P&L
            week_start = bar['timestamp'] - timedelta(days=7)
            self.weekly_pnl.append((week_start, self.current_week_pnl))
            self.current_week_pnl = 0.0

        self.last_week = current_week

    async def _generate_strategy_signal(self, strategy_func: callable,
                                      data_window: pd.DataFrame,
                                      params: Dict) -> Tuple[SwingSignal, SwingPattern]:
        """Generate trading signal and pattern from strategy function"""
        try:
            result = strategy_func(data_window, params)

            # Handle different return formats
            if isinstance(result, tuple):
                signal, pattern = result
            else:
                signal = result
                pattern = SwingPattern.CONTINUATION  # Default pattern

            # Convert string signals to enum
            if isinstance(signal, str):
                signal = SwingSignal(signal.lower())
            if isinstance(pattern, str):
                pattern = SwingPattern(pattern.lower())

            return signal, pattern

        except Exception as e:
            logger.warning(f"Strategy signal generation failed: {e}")
            return SwingSignal.HOLD, SwingPattern.CONTINUATION

    async def _process_swing_signal(self, signal: SwingSignal, pattern: SwingPattern,
                                  bar: pd.Series, symbol: str):
        """Process swing trading signal and execute trades"""

        if signal == SwingSignal.BUY and len(self.current_positions) < self.max_positions:
            await self._open_position("long", bar, signal.value, pattern, symbol)

        elif signal == SwingSignal.SELL and len(self.current_positions) < self.max_positions:
            await self._open_position("short", bar, signal.value, pattern, symbol)

        elif signal == SwingSignal.CLOSE_LONG:
            await self._close_positions_by_direction("long", bar, signal.value)

        elif signal == SwingSignal.CLOSE_SHORT:
            await self._close_positions_by_direction("short", bar, signal.value)

        elif signal == SwingSignal.ADD_TO_POSITION:
            await self._add_to_positions(bar, pattern, symbol)

        elif signal == SwingSignal.REDUCE_POSITION:
            await self._reduce_positions(bar)

    async def _open_position(self, direction: str, bar: pd.Series,
                           signal: str, pattern: SwingPattern, symbol: str):
        """Open new swing trading position"""

        # Calculate position size based on ATR and risk management
        atr = bar['atr']
        risk_amount = self.current_balance * self.max_risk_per_trade

        # Stop loss distance based on ATR
        stop_distance = atr * 2.0  # 2 ATR stop loss

        # Position size calculation
        pip_value = 0.0001
        position_size = risk_amount / (stop_distance / pip_value * 10)  # Simplified
        position_size = min(position_size, 2.0)  # Max 2 lots for swing trading

        # Determine entry price and stop loss
        entry_price = bar['close']
        if direction == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2.0)  # 1:2 risk/reward
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2.0)

        # Calculate costs
        commission = self.commission_per_lot * position_size

        position = {
            'id': len(self.current_positions),
            'direction': direction,
            'entry_time': bar['timestamp'],
            'entry_price': entry_price,
            'quantity': position_size,
            'entry_signal': signal,
            'pattern': pattern,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'commission': commission,
            'symbol': symbol,
            'max_favorable_excursion': 0.0,
            'max_adverse_excursion': 0.0,
            'total_swap': 0.0
        }

        self.current_positions.append(position)
        logger.debug(f"Opened {direction} {pattern.value} position at {entry_price} on {bar['timestamp']}")

    async def _close_position(self, position: Dict, bar: pd.Series, signal: str):
        """Close specific swing position"""

        exit_price = bar['close']

        # Calculate P&L
        if position['direction'] == "long":
            pnl = (exit_price - position['entry_price']) * position['quantity'] * 100000
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity'] * 100000

        # Calculate total swap cost
        days_held = (bar['timestamp'] - position['entry_time']).days
        total_swap = self.swap_per_lot_per_day * position['quantity'] * days_held
        position['total_swap'] = total_swap

        # Net P&L
        net_pnl = pnl - position['commission'] - total_swap

        # Calculate pips
        pip_value = 0.0001
        if position['direction'] == "long":
            pips = (exit_price - position['entry_price']) / pip_value
        else:
            pips = (position['entry_price'] - exit_price) / pip_value

        # Calculate duration and risk/reward ratio
        duration = (bar['timestamp'] - position['entry_time']).days

        # Risk/reward calculation
        risk = abs(position['entry_price'] - position['stop_loss'])
        reward = abs(exit_price - position['entry_price'])
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        # Create trade record
        trade = SwingTrade(
            entry_time=position['entry_time'],
            exit_time=bar['timestamp'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            direction=position['direction'],
            entry_signal=position['entry_signal'],
            exit_signal=signal,
            pattern=position['pattern'],
            pnl=net_pnl,
            pips=pips,
            duration_days=duration,
            max_favorable_excursion=position['max_favorable_excursion'],
            max_adverse_excursion=position['max_adverse_excursion'],
            commission=position['commission'],
            swap_total=total_swap,
            risk_reward_ratio=risk_reward_ratio
        )

        self.trades.append(trade)
        self.current_balance += net_pnl
        self.current_week_pnl += net_pnl

        # Remove position from current positions
        self.current_positions.remove(position)

        logger.debug(f"Closed {position['pattern'].value} position: {net_pnl:.2f} PnL, {pips:.1f} pips, {duration:.0f} days")

    async def _close_positions_by_direction(self, direction: str, bar: pd.Series, signal: str):
        """Close all positions in specified direction"""
        positions_to_close = [pos for pos in self.current_positions if pos['direction'] == direction]

        for position in positions_to_close:
            await self._close_position(position, bar, signal)

    async def _add_to_positions(self, bar: pd.Series, pattern: SwingPattern, symbol: str):
        """Add to existing positions (pyramiding)"""
        if self.current_positions and len(self.current_positions) < self.max_positions:
            # Add to position in same direction as most profitable current position
            profitable_positions = [pos for pos in self.current_positions
                                  if self._calculate_unrealized_pnl(pos, bar) > 0]

            if profitable_positions:
                best_position = max(profitable_positions,
                                  key=lambda x: self._calculate_unrealized_pnl(x, bar))
                await self._open_position(best_position['direction'], bar, "add_to_position", pattern, symbol)

    async def _reduce_positions(self, bar: pd.Series):
        """Reduce position sizes (partial close)"""
        if len(self.current_positions) > 1:
            # Close least profitable position
            worst_position = min(self.current_positions,
                               key=lambda x: self._calculate_unrealized_pnl(x, bar))
            await self._close_position(worst_position, bar, "reduce_position")

    def _calculate_unrealized_pnl(self, position: Dict, bar: pd.Series) -> float:
        """Calculate unrealized P&L for a position"""
        current_price = bar['close']

        if position['direction'] == "long":
            return (current_price - position['entry_price']) * position['quantity'] * 100000
        else:
            return (position['entry_price'] - current_price) * position['quantity'] * 100000

    async def _update_positions(self, bar: pd.Series):
        """Update position metrics and check for stop losses/take profits"""
        current_price = bar['close']
        positions_to_close = []

        for position in self.current_positions:
            # Update max favorable/adverse excursion
            unrealized_pnl = self._calculate_unrealized_pnl(position, bar)

            if unrealized_pnl > position['max_favorable_excursion']:
                position['max_favorable_excursion'] = unrealized_pnl
            elif unrealized_pnl < 0 and abs(unrealized_pnl) > position['max_adverse_excursion']:
                position['max_adverse_excursion'] = abs(unrealized_pnl)

            # Check stop loss and take profit
            if position['direction'] == "long":
                if current_price <= position['stop_loss']:
                    positions_to_close.append((position, "stop_loss"))
                elif current_price >= position['take_profit']:
                    positions_to_close.append((position, "take_profit"))
            else:
                if current_price >= position['stop_loss']:
                    positions_to_close.append((position, "stop_loss"))
                elif current_price <= position['take_profit']:
                    positions_to_close.append((position, "take_profit"))

        # Close positions that hit stop loss or take profit
        for position, reason in positions_to_close:
            await self._close_position(position, bar, reason)

    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current balance"""
        current_equity = self.current_balance

        # Add unrealized P&L from open positions
        # This would need current market price - simplified for now
        current_equity += 0  # Placeholder for unrealized P&L

        self.equity_curve.append((timestamp, current_equity))

    def _update_performance_tracking(self):
        """Update performance tracking metrics"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        self.drawdown_curve.append((datetime.now(), current_drawdown))

    async def _calculate_swing_metrics(self, data: pd.DataFrame) -> SwingTradingMetrics:
        """Calculate comprehensive swing trading metrics"""

        if not self.trades:
            return SwingTradingMetrics(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                total_pnl=0.0, total_pips=0.0, avg_trade_duration=0.0,
                avg_pips_per_trade=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
                profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                recovery_factor=0.0, trades_per_week=0.0, avg_risk_reward_ratio=0.0,
                best_pattern=SwingPattern.CONTINUATION, worst_pattern=SwingPattern.CONTINUATION,
                pattern_metrics={}, weekly_metrics=[]
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
        avg_trade_duration = np.mean([trade.duration_days for trade in self.trades])

        # Risk/reward metrics
        valid_rr_trades = [trade for trade in self.trades if trade.risk_reward_ratio is not None]
        avg_risk_reward_ratio = np.mean([trade.risk_reward_ratio for trade in valid_rr_trades]) if valid_rr_trades else 0.0

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
        total_weeks = (data.iloc[-1]['timestamp'] - data.iloc[0]['timestamp']).days / 7
        trades_per_week = total_trades / total_weeks if total_weeks > 0 else 0.0

        # Pattern analysis
        pattern_metrics = self._calculate_pattern_metrics()
        best_pattern = max(pattern_metrics.keys(), key=lambda x: pattern_metrics[x]['total_pnl']) if pattern_metrics else SwingPattern.CONTINUATION
        worst_pattern = min(pattern_metrics.keys(), key=lambda x: pattern_metrics[x]['total_pnl']) if pattern_metrics else SwingPattern.CONTINUATION

        # Weekly analysis
        weekly_metrics = self._calculate_weekly_metrics()

        return SwingTradingMetrics(
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
            trades_per_week=trades_per_week,
            avg_risk_reward_ratio=avg_risk_reward_ratio,
            best_pattern=best_pattern,
            worst_pattern=worst_pattern,
            pattern_metrics=pattern_metrics,
            weekly_metrics=weekly_metrics
        )

    def _calculate_pattern_metrics(self) -> Dict[SwingPattern, Dict]:
        """Calculate metrics for each swing pattern"""
        pattern_trades = {}

        # Group trades by pattern
        for trade in self.trades:
            pattern = trade.pattern
            if pattern not in pattern_trades:
                pattern_trades[pattern] = []
            pattern_trades[pattern].append(trade)

        # Calculate metrics for each pattern
        pattern_metrics = {}
        for pattern, trades in pattern_trades.items():
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.pnl > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            total_pnl = sum(trade.pnl for trade in trades)
            total_pips = sum(trade.pips for trade in trades)
            avg_duration = np.mean([trade.duration_days for trade in trades])
            avg_rr = np.mean([trade.risk_reward_ratio for trade in trades if trade.risk_reward_ratio is not None])

            # Profit factor
            winning_pnl = sum(trade.pnl for trade in trades if trade.pnl > 0)
            losing_pnl = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

            pattern_metrics[pattern] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pips': total_pips,
                'avg_duration': avg_duration,
                'avg_risk_reward': avg_rr,
                'profit_factor': profit_factor
            }

        return pattern_metrics

    def _calculate_weekly_metrics(self) -> List[WeeklyMetrics]:
        """Calculate weekly performance metrics"""
        weekly_trades = {}

        # Group trades by week
        for trade in self.trades:
            week_key = (trade.entry_time.year, trade.entry_time.isocalendar().week)
            if week_key not in weekly_trades:
                weekly_trades[week_key] = []
            weekly_trades[week_key].append(trade)

        # Calculate metrics for each week
        weekly_metrics = []
        for (year, week), trades in weekly_trades.items():
            week_start = datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")

            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.pnl > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            total_pnl = sum(trade.pnl for trade in trades)
            total_pips = sum(trade.pips for trade in trades)

            # Calculate weekly drawdown
            weekly_equity = []
            running_pnl = 0
            for trade in sorted(trades, key=lambda x: x.entry_time):
                running_pnl += trade.pnl
                weekly_equity.append(running_pnl)

            peak = 0
            max_dd = 0
            for equity in weekly_equity:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / abs(peak) if peak != 0 else 0
                if dd > max_dd:
                    max_dd = dd

            # Profit factor
            winning_pnl = sum(trade.pnl for trade in trades if trade.pnl > 0)
            losing_pnl = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

            weekly_metrics.append(WeeklyMetrics(
                week_start=week_start,
                total_trades=total_trades,
                winning_trades=winning_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pips=total_pips,
                max_drawdown=max_dd,
                profit_factor=profit_factor
            ))

        return weekly_metrics

    def _calculate_max_consecutive(self, trades: List[SwingTrade], winning: bool) -> int:
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
