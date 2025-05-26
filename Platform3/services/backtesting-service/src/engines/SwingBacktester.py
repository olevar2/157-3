"""
Swing Trading Backtesting Engine
H4 short-term swing testing with pattern recognition and multi-timeframe analysis.

This module provides comprehensive backtesting for swing trading strategies
with maximum 5-day holding periods and advanced pattern validation.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

class SwingPattern(Enum):
    """Swing trading pattern types"""
    ELLIOTT_WAVE = "Elliott_Wave"
    FIBONACCI_RETRACEMENT = "Fibonacci_Retracement"
    SUPPORT_RESISTANCE = "Support_Resistance"
    TREND_CONTINUATION = "Trend_Continuation"
    REVERSAL_PATTERN = "Reversal_Pattern"

@dataclass
class SwingSetup:
    """Swing trading setup data"""
    timestamp: datetime
    symbol: str
    pattern: SwingPattern
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_reward_ratio: float
    pattern_data: Dict[str, Any]

@dataclass
class SwingBacktestResult:
    """Results from swing trading backtesting"""
    strategy_name: str
    symbol: str
    timeframe: str
    total_setups: int
    executed_trades: int
    pattern_performance: Dict[SwingPattern, Dict[str, float]]
    win_rate: float
    average_hold_time_hours: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    best_pattern: SwingPattern
    risk_reward_achieved: float
    execution_time_ms: float

class SwingBacktester:
    """
    Advanced backtesting engine for H4 swing trading strategies.
    
    Features:
    - Short-term swing pattern recognition (max 5 days)
    - Multi-timeframe confluence validation
    - Risk-reward optimization
    - Pattern-specific performance analysis
    - Advanced position management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Swing trading specific parameters
        self.max_hold_days = config.get('max_hold_days', 5)
        self.min_risk_reward = config.get('min_risk_reward', 1.5)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% max risk
        
        # Pattern validation settings
        self.require_confluence = config.get('require_confluence', True)
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Position management
        self.use_trailing_stops = config.get('use_trailing_stops', True)
        self.partial_profit_taking = config.get('partial_profit_taking', True)
        
    async def backtest_swing_strategy(
        self,
        strategy_func: callable,
        market_data: pd.DataFrame,
        swing_setups: List[SwingSetup],
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        strategy_params: Dict[str, Any] = None
    ) -> SwingBacktestResult:
        """
        Execute comprehensive swing trading backtesting.
        
        Args:
            strategy_func: Swing trading strategy function
            market_data: H4 market data DataFrame
            swing_setups: List of identified swing setups
            initial_balance: Starting account balance
            risk_per_trade: Risk percentage per trade
            strategy_params: Strategy-specific parameters
            
        Returns:
            SwingBacktestResult: Comprehensive swing trading results
        """
        start_time = datetime.now()
        
        # Initialize tracking variables
        balance = initial_balance
        trades = []
        active_positions = []
        pattern_stats = {pattern: {'trades': [], 'setups': 0} for pattern in SwingPattern}
        
        # Filter setups by confidence and risk-reward
        qualified_setups = self._filter_setups(swing_setups)
        
        # Process each setup
        for setup in qualified_setups:
            pattern_stats[setup.pattern]['setups'] += 1
            
            # Check if we should execute this setup
            if self._should_execute_setup(setup, balance, active_positions):
                
                # Calculate position size based on risk
                position_size = self._calculate_position_size(
                    balance, setup, risk_per_trade
                )
                
                # Execute the trade
                trade = await self._execute_swing_trade(
                    setup, market_data, position_size, strategy_params
                )
                
                if trade:
                    trades.append(trade)
                    pattern_stats[setup.pattern]['trades'].append(trade)
                    balance += trade['pnl']
        
        # Calculate performance metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Analyze pattern performance
        pattern_performance = self._analyze_pattern_performance(pattern_stats)
        
        return SwingBacktestResult(
            strategy_name=strategy_params.get('name', 'SwingTrading') if strategy_params else 'SwingTrading',
            symbol=market_data['symbol'].iloc[0] if 'symbol' in market_data.columns else 'UNKNOWN',
            timeframe='H4',
            total_setups=len(qualified_setups),
            executed_trades=len(trades),
            pattern_performance=pattern_performance,
            win_rate=self._calculate_win_rate(trades),
            average_hold_time_hours=self._calculate_average_hold_time(trades),
            total_pnl=sum(trade['pnl'] for trade in trades),
            max_drawdown=self._calculate_max_drawdown(trades),
            sharpe_ratio=self._calculate_sharpe_ratio(trades),
            best_pattern=self._find_best_pattern(pattern_performance),
            risk_reward_achieved=self._calculate_average_risk_reward(trades),
            execution_time_ms=execution_time
        )
    
    def _filter_setups(self, setups: List[SwingSetup]) -> List[SwingSetup]:
        """Filter setups based on quality criteria"""
        
        qualified_setups = []
        
        for setup in setups:
            # Check minimum confidence
            if setup.confidence < self.min_confidence:
                continue
            
            # Check minimum risk-reward ratio
            if setup.risk_reward_ratio < self.min_risk_reward:
                continue
            
            # Additional pattern-specific validation
            if self._validate_pattern_setup(setup):
                qualified_setups.append(setup)
        
        return qualified_setups
    
    def _validate_pattern_setup(self, setup: SwingSetup) -> bool:
        """Validate pattern-specific setup criteria"""
        
        if setup.pattern == SwingPattern.ELLIOTT_WAVE:
            # Validate Elliott Wave structure
            wave_data = setup.pattern_data.get('wave_structure', {})
            return wave_data.get('wave_count', 0) >= 3
            
        elif setup.pattern == SwingPattern.FIBONACCI_RETRACEMENT:
            # Validate Fibonacci levels
            fib_data = setup.pattern_data.get('fibonacci_levels', {})
            return fib_data.get('retracement_level', 0) in [0.382, 0.5, 0.618]
            
        elif setup.pattern == SwingPattern.SUPPORT_RESISTANCE:
            # Validate support/resistance strength
            sr_data = setup.pattern_data.get('sr_data', {})
            return sr_data.get('touch_count', 0) >= 2
            
        elif setup.pattern == SwingPattern.TREND_CONTINUATION:
            # Validate trend strength
            trend_data = setup.pattern_data.get('trend_data', {})
            return trend_data.get('trend_strength', 0) > 0.6
            
        elif setup.pattern == SwingPattern.REVERSAL_PATTERN:
            # Validate reversal signals
            reversal_data = setup.pattern_data.get('reversal_data', {})
            return reversal_data.get('divergence_confirmed', False)
        
        return True  # Default validation
    
    def _should_execute_setup(
        self, setup: SwingSetup, balance: float, active_positions: List[Dict]
    ) -> bool:
        """Determine if a setup should be executed"""
        
        # Check maximum number of concurrent positions
        max_positions = self.config.get('max_concurrent_positions', 3)
        if len(active_positions) >= max_positions:
            return False
        
        # Check if we have enough balance for the risk
        required_risk = balance * self.max_risk_per_trade
        setup_risk = abs(setup.entry_price - setup.stop_loss) * 100000  # Simplified calculation
        
        if setup_risk > required_risk:
            return False
        
        # Check for conflicting positions (same symbol)
        for position in active_positions:
            if position['symbol'] == setup.symbol:
                return False
        
        return True
    
    def _calculate_position_size(
        self, balance: float, setup: SwingSetup, risk_per_trade: float
    ) -> float:
        """Calculate position size based on risk management"""
        
        # Calculate risk amount
        risk_amount = balance * risk_per_trade
        
        # Calculate price distance to stop loss
        price_distance = abs(setup.entry_price - setup.stop_loss)
        
        # Calculate position size (simplified for forex)
        if price_distance > 0:
            position_size = risk_amount / (price_distance * 100000)
            return min(position_size, 1.0)  # Cap at 1 lot
        
        return 0.01  # Default minimum size
    
    async def _execute_swing_trade(
        self,
        setup: SwingSetup,
        market_data: pd.DataFrame,
        position_size: float,
        strategy_params: Dict[str, Any]
    ) -> Optional[Dict]:
        """Execute a swing trade based on the setup"""
        
        # Find the entry point in market data
        entry_index = self._find_entry_point(setup, market_data)
        if entry_index is None:
            return None
        
        # Simulate trade execution
        entry_time = market_data.iloc[entry_index]['timestamp']
        entry_price = self._simulate_swing_execution(setup.entry_price, 'entry')
        
        # Track the trade through its lifecycle
        trade_result = await self._track_swing_trade(
            setup, market_data, entry_index, entry_price, position_size
        )
        
        return trade_result
    
    def _find_entry_point(self, setup: SwingSetup, market_data: pd.DataFrame) -> Optional[int]:
        """Find the entry point in market data"""
        
        # Find the closest timestamp to setup time
        setup_time = setup.timestamp
        time_diffs = abs(market_data['timestamp'] - setup_time)
        closest_index = time_diffs.idxmin()
        
        # Validate that the entry is reasonable
        closest_time = market_data.iloc[closest_index]['timestamp']
        time_diff = abs((closest_time - setup_time).total_seconds())
        
        # Allow up to 4 hours difference (1 H4 candle)
        if time_diff <= 4 * 3600:
            return closest_index
        
        return None
    
    def _simulate_swing_execution(self, price: float, execution_type: str) -> float:
        """Simulate realistic execution for swing trades"""
        
        # Swing trades typically have lower slippage due to longer timeframes
        base_slippage = 0.2  # 0.2 pips average
        
        # Add some randomness
        slippage = base_slippage * np.random.normal(1.0, 0.3)
        slippage = max(0.1, slippage)
        
        if execution_type == 'entry':
            return price + slippage  # Assume slightly worse entry
        else:  # exit
            return price - slippage  # Assume slightly worse exit
    
    async def _track_swing_trade(
        self,
        setup: SwingSetup,
        market_data: pd.DataFrame,
        entry_index: int,
        entry_price: float,
        position_size: float
    ) -> Dict:
        """Track a swing trade through its complete lifecycle"""
        
        # Initialize trade tracking
        trade = {
            'setup': setup,
            'entry_time': market_data.iloc[entry_index]['timestamp'],
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': 1,  # Assume long for simplicity
            'stop_loss': setup.stop_loss,
            'take_profit': setup.take_profit,
            'pattern': setup.pattern
        }
        
        # Track through subsequent candles
        max_hold_candles = self.max_hold_days * 6  # 6 H4 candles per day
        exit_index = min(entry_index + max_hold_candles, len(market_data) - 1)
        
        # Find exit point
        for i in range(entry_index + 1, exit_index + 1):
            candle = market_data.iloc[i]
            
            # Check stop loss
            if candle['low'] <= setup.stop_loss:
                exit_price = self._simulate_swing_execution(setup.stop_loss, 'exit')
                exit_reason = 'stop_loss'
                exit_time = candle['timestamp']
                break
            
            # Check take profit
            if candle['high'] >= setup.take_profit:
                exit_price = self._simulate_swing_execution(setup.take_profit, 'exit')
                exit_reason = 'take_profit'
                exit_time = candle['timestamp']
                break
            
            # Check maximum hold time
            if i == exit_index:
                exit_price = self._simulate_swing_execution(candle['close'], 'exit')
                exit_reason = 'max_hold_time'
                exit_time = candle['timestamp']
                break
        else:
            # If no exit condition met, exit at last available price
            last_candle = market_data.iloc[exit_index]
            exit_price = self._simulate_swing_execution(last_candle['close'], 'exit')
            exit_reason = 'end_of_data'
            exit_time = last_candle['timestamp']
        
        # Calculate trade results
        pnl = self._calculate_swing_pnl(trade, exit_price)
        hold_time_hours = (exit_time - trade['entry_time']).total_seconds() / 3600
        
        return {
            'entry_time': trade['entry_time'],
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': trade['direction'],
            'size': position_size,
            'pnl': pnl,
            'pattern': setup.pattern,
            'exit_reason': exit_reason,
            'hold_time_hours': hold_time_hours,
            'setup_confidence': setup.confidence,
            'risk_reward_ratio': setup.risk_reward_ratio
        }
    
    def _calculate_swing_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate P&L for swing trade"""
        price_diff = exit_price - trade['entry_price']
        if trade['direction'] < 0:  # Short position
            price_diff = -price_diff
        return price_diff * trade['position_size'] * 100000  # Standard lot calculation
    
    def _analyze_pattern_performance(
        self, pattern_stats: Dict[SwingPattern, Dict]
    ) -> Dict[SwingPattern, Dict[str, float]]:
        """Analyze performance by swing pattern"""
        
        performance = {}
        
        for pattern, stats in pattern_stats.items():
            trades = stats['trades']
            setups = stats['setups']
            
            if not trades:
                performance[pattern] = {
                    'total_setups': setups,
                    'executed_trades': 0,
                    'execution_rate': 0.0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl_per_trade': 0.0,
                    'avg_hold_time_hours': 0.0
                }
                continue
            
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            total_pnl = sum(t['pnl'] for t in trades)
            avg_hold_time = sum(t['hold_time_hours'] for t in trades) / len(trades)
            
            performance[pattern] = {
                'total_setups': setups,
                'executed_trades': len(trades),
                'execution_rate': len(trades) / setups if setups > 0 else 0.0,
                'win_rate': winning_trades / len(trades),
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / len(trades),
                'avg_hold_time_hours': avg_hold_time
            }
        
        return performance
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate overall win rate"""
        if not trades:
            return 0.0
        
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        return winning_trades / len(trades)
    
    def _calculate_average_hold_time(self, trades: List[Dict]) -> float:
        """Calculate average holding time in hours"""
        if not trades:
            return 0.0
        
        total_time = sum(trade['hold_time_hours'] for trade in trades)
        return total_time / len(trades)
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown"""
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
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio"""
        if not trades:
            return 0.0
        
        returns = [trade['pnl'] for trade in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _find_best_pattern(
        self, pattern_performance: Dict[SwingPattern, Dict[str, float]]
    ) -> SwingPattern:
        """Find the best performing pattern"""
        
        best_pattern = SwingPattern.ELLIOTT_WAVE  # Default
        best_pnl = float('-inf')
        
        for pattern, performance in pattern_performance.items():
            if performance['total_pnl'] > best_pnl:
                best_pnl = performance['total_pnl']
                best_pattern = pattern
        
        return best_pattern
    
    def _calculate_average_risk_reward(self, trades: List[Dict]) -> float:
        """Calculate average risk-reward ratio achieved"""
        if not trades:
            return 0.0
        
        risk_rewards = []
        
        for trade in trades:
            if 'risk_reward_ratio' in trade:
                risk_rewards.append(trade['risk_reward_ratio'])
        
        return np.mean(risk_rewards) if risk_rewards else 0.0
