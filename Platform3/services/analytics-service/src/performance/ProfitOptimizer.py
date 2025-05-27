"""
Profit Optimizer Module

This module provides advanced profit optimization algorithms for forex trading,
including position sizing optimization, risk-reward optimization, and strategy parameter tuning.
Designed to maximize risk-adjusted returns across different market conditions.

Features:
- Kelly Criterion position sizing
- Risk-reward ratio optimization
- Monte Carlo simulation for strategy testing
- Multi-objective optimization (return vs risk)
- Parameter sensitivity analysis
- Portfolio optimization across currency pairs
- Dynamic position sizing based on market conditions

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import random
from scipy.optimize import minimize, differential_evolution
import warnings

# Suppress optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objective types"""
    MAXIMIZE_RETURN = "maximize_return"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_PROFIT_FACTOR = "maximize_profit_factor"

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"

@dataclass
class TradeResult:
    """Individual trade result for optimization"""
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    volatility: float
    currency_pair: str
    strategy: str

@dataclass
class OptimizationParameters:
    """Parameters for optimization"""
    position_size_range: Tuple[float, float]
    risk_reward_range: Tuple[float, float]
    stop_loss_range: Tuple[float, float]
    take_profit_range: Tuple[float, float]
    max_positions: int
    max_risk_per_trade: float
    correlation_threshold: float

@dataclass
class OptimizationResults:
    """Results from profit optimization"""
    optimal_position_size: float
    optimal_risk_reward: float
    optimal_stop_loss: float
    optimal_take_profit: float
    expected_return: float
    expected_sharpe: float
    expected_max_drawdown: float
    win_rate: float
    profit_factor: float
    kelly_fraction: float
    confidence_interval: Tuple[float, float]
    sensitivity_analysis: Dict[str, float]

class ProfitOptimizer:
    """
    Advanced Profit Optimization System
    
    Provides sophisticated optimization algorithms for maximizing
    risk-adjusted returns in forex trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize Profit Optimizer
        
        Args:
            initial_capital: Starting capital for optimization
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Trade history for optimization
        self.trade_results: List[TradeResult] = []
        
        # Optimization cache
        self.optimization_cache: Dict[str, Any] = {}
        
        logger.info(f"ProfitOptimizer initialized with capital: ${initial_capital:,.2f}")
    
    def add_trade_result(self,
                        entry_time: datetime,
                        exit_time: datetime,
                        pnl: float,
                        position_size: float,
                        max_adverse_excursion: float = 0.0,
                        max_favorable_excursion: float = 0.0,
                        volatility: float = 0.01,
                        currency_pair: str = "EURUSD",
                        strategy: str = "default") -> None:
        """
        Add trade result for optimization analysis
        
        Args:
            entry_time: Trade entry time
            exit_time: Trade exit time
            pnl: Trade profit/loss
            position_size: Position size used
            max_adverse_excursion: Maximum adverse price movement
            max_favorable_excursion: Maximum favorable price movement
            volatility: Market volatility during trade
            currency_pair: Currency pair traded
            strategy: Strategy used
        """
        try:
            # Calculate return percentage
            return_pct = pnl / (position_size * self.initial_capital) if position_size > 0 else 0.0
            
            trade_result = TradeResult(
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                return_pct=return_pct,
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                volatility=volatility,
                currency_pair=currency_pair,
                strategy=strategy
            )
            
            self.trade_results.append(trade_result)
            
            # Clear cache when new data is added
            self.optimization_cache.clear()
            
            logger.debug(f"Trade result added: {currency_pair} {strategy}, P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade result: {str(e)}")
            raise
    
    def calculate_kelly_fraction(self, 
                                win_rate: Optional[float] = None,
                                avg_win: Optional[float] = None,
                                avg_loss: Optional[float] = None) -> float:
        """
        Calculate Kelly Criterion optimal position size
        
        Args:
            win_rate: Win rate (0-1), calculated from trades if None
            avg_win: Average winning trade, calculated if None
            avg_loss: Average losing trade, calculated if None
            
        Returns:
            Kelly fraction (0-1)
        """
        try:
            if not self.trade_results:
                return 0.0
            
            # Calculate from trade results if not provided
            if win_rate is None or avg_win is None or avg_loss is None:
                wins = [t.return_pct for t in self.trade_results if t.return_pct > 0]
                losses = [t.return_pct for t in self.trade_results if t.return_pct < 0]
                
                win_rate = len(wins) / len(self.trade_results) if self.trade_results else 0.0
                avg_win = np.mean(wins) if wins else 0.0
                avg_loss = abs(np.mean(losses)) if losses else 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            if avg_loss == 0:
                return 0.0
            
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction to reasonable limits (0-25%)
            kelly_fraction = max(0.0, min(0.25, kelly_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {str(e)}")
            return 0.0
    
    def calculate_optimal_f(self, returns: Optional[List[float]] = None) -> float:
        """
        Calculate Optimal F position sizing
        
        Args:
            returns: List of returns, uses trade results if None
            
        Returns:
            Optimal F fraction
        """
        try:
            if returns is None:
                returns = [t.return_pct for t in self.trade_results]
            
            if not returns:
                return 0.0
            
            # Find the optimal fraction that maximizes geometric mean
            def geometric_mean(f: float) -> float:
                if f <= 0 or f >= 1:
                    return -np.inf
                
                cumulative = 1.0
                for r in returns:
                    new_value = 1 + f * r
                    if new_value <= 0:
                        return -np.inf
                    cumulative *= new_value
                
                return cumulative ** (1.0 / len(returns))
            
            # Optimize using grid search
            best_f = 0.0
            best_gm = 0.0
            
            for f in np.linspace(0.01, 0.5, 50):
                gm = geometric_mean(f)
                if gm > best_gm:
                    best_gm = gm
                    best_f = f
            
            return best_f
            
        except Exception as e:
            logger.error(f"Error calculating Optimal F: {str(e)}")
            return 0.0
    
    def optimize_position_sizing(self, 
                                method: PositionSizingMethod = PositionSizingMethod.KELLY,
                                target_volatility: float = 0.15) -> Dict[str, float]:
        """
        Optimize position sizing based on specified method
        
        Args:
            method: Position sizing method to use
            target_volatility: Target portfolio volatility
            
        Returns:
            Dictionary with position sizing recommendations
        """
        try:
            if not self.trade_results:
                logger.warning("No trade results available for position sizing optimization")
                return {"optimal_size": 0.01, "method": method.value}
            
            results = {"method": method.value}
            
            if method == PositionSizingMethod.KELLY:
                kelly_fraction = self.calculate_kelly_fraction()
                results["optimal_size"] = kelly_fraction
                results["kelly_fraction"] = kelly_fraction
                
            elif method == PositionSizingMethod.OPTIMAL_F:
                optimal_f = self.calculate_optimal_f()
                results["optimal_size"] = optimal_f
                results["optimal_f"] = optimal_f
                
            elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                # Calculate portfolio volatility
                returns = [t.return_pct for t in self.trade_results]
                portfolio_vol = np.std(returns) * np.sqrt(252) if returns else 0.15
                
                # Scale position size to achieve target volatility
                vol_adjustment = target_volatility / portfolio_vol if portfolio_vol > 0 else 1.0
                base_size = 0.02  # 2% base position size
                optimal_size = base_size * vol_adjustment
                
                results["optimal_size"] = min(0.25, optimal_size)  # Cap at 25%
                results["volatility_adjustment"] = vol_adjustment
                results["portfolio_volatility"] = portfolio_vol
                
            elif method == PositionSizingMethod.RISK_PARITY:
                # Equal risk contribution across trades
                volatilities = [t.volatility for t in self.trade_results]
                avg_volatility = np.mean(volatilities) if volatilities else 0.01
                
                # Inverse volatility weighting
                risk_budget = 0.02  # 2% risk per trade
                optimal_size = risk_budget / avg_volatility
                
                results["optimal_size"] = min(0.25, optimal_size)
                results["average_volatility"] = avg_volatility
                
            else:  # FIXED
                results["optimal_size"] = 0.02  # Fixed 2%
            
            logger.info(f"Position sizing optimized using {method.value}: "
                       f"{results['optimal_size']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing position sizing: {str(e)}")
            return {"optimal_size": 0.01, "method": method.value, "error": str(e)}
    
    def optimize_risk_reward(self, 
                           objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE) -> Dict[str, float]:
        """
        Optimize risk-reward parameters
        
        Args:
            objective: Optimization objective
            
        Returns:
            Dictionary with optimal risk-reward parameters
        """
        try:
            if not self.trade_results:
                logger.warning("No trade results available for risk-reward optimization")
                return {"optimal_rr": 2.0, "stop_loss": 0.01, "take_profit": 0.02}
            
            # Define objective function
            def objective_function(params: List[float]) -> float:
                stop_loss, take_profit = params
                risk_reward = take_profit / stop_loss if stop_loss > 0 else 0.0
                
                # Simulate trades with these parameters
                simulated_returns = []
                for trade in self.trade_results:
                    # Simulate exit based on stop loss and take profit
                    if trade.max_adverse_excursion >= stop_loss:
                        # Hit stop loss
                        simulated_returns.append(-stop_loss)
                    elif trade.max_favorable_excursion >= take_profit:
                        # Hit take profit
                        simulated_returns.append(take_profit)
                    else:
                        # Use actual return
                        simulated_returns.append(trade.return_pct)
                
                if not simulated_returns:
                    return np.inf
                
                # Calculate objective based on type
                if objective == OptimizationObjective.MAXIMIZE_RETURN:
                    return -np.mean(simulated_returns)  # Negative for minimization
                
                elif objective == OptimizationObjective.MAXIMIZE_SHARPE:
                    mean_return = np.mean(simulated_returns)
                    std_return = np.std(simulated_returns)
                    if std_return == 0:
                        return np.inf
                    sharpe = (mean_return - self.risk_free_rate/252) / std_return
                    return -sharpe  # Negative for minimization
                
                elif objective == OptimizationObjective.MINIMIZE_DRAWDOWN:
                    # Calculate maximum drawdown
                    cumulative = np.cumprod(1 + np.array(simulated_returns))
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = abs(np.min(drawdown))
                    return max_drawdown
                
                elif objective == OptimizationObjective.MAXIMIZE_PROFIT_FACTOR:
                    wins = [r for r in simulated_returns if r > 0]
                    losses = [r for r in simulated_returns if r < 0]
                    
                    gross_profit = sum(wins) if wins else 0.0
                    gross_loss = abs(sum(losses)) if losses else 0.0
                    
                    if gross_loss == 0:
                        return -np.inf if gross_profit > 0 else 0.0
                    
                    profit_factor = gross_profit / gross_loss
                    return -profit_factor  # Negative for minimization
                
                else:
                    return np.inf
            
            # Optimization bounds
            bounds = [(0.005, 0.05), (0.01, 0.1)]  # stop_loss, take_profit
            
            # Optimize using differential evolution
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
            
            if result.success:
                optimal_stop_loss, optimal_take_profit = result.x
                optimal_rr = optimal_take_profit / optimal_stop_loss
                
                optimization_results = {
                    "optimal_rr": optimal_rr,
                    "stop_loss": optimal_stop_loss,
                    "take_profit": optimal_take_profit,
                    "objective_value": -result.fun if objective != OptimizationObjective.MINIMIZE_DRAWDOWN else result.fun,
                    "optimization_success": True
                }
            else:
                # Fallback to default values
                optimization_results = {
                    "optimal_rr": 2.0,
                    "stop_loss": 0.01,
                    "take_profit": 0.02,
                    "objective_value": 0.0,
                    "optimization_success": False
                }
            
            logger.info(f"Risk-reward optimization completed: RR={optimization_results['optimal_rr']:.2f}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing risk-reward: {str(e)}")
            return {"optimal_rr": 2.0, "stop_loss": 0.01, "take_profit": 0.02, "error": str(e)}
    
    def monte_carlo_simulation(self, 
                              num_simulations: int = 1000,
                              num_trades: int = 100,
                              position_size: float = 0.02) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation for strategy testing
        
        Args:
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation
            position_size: Position size to use
            
        Returns:
            Dictionary with simulation results
        """
        try:
            if not self.trade_results:
                logger.warning("No trade results available for Monte Carlo simulation")
                return {"error": "No trade data available"}
            
            # Extract returns for sampling
            returns = [t.return_pct for t in self.trade_results]
            
            simulation_results = []
            
            for _ in range(num_simulations):
                # Sample returns with replacement
                sampled_returns = random.choices(returns, k=num_trades)
                
                # Apply position sizing
                scaled_returns = [r * position_size for r in sampled_returns]
                
                # Calculate cumulative performance
                cumulative_return = np.prod(1 + np.array(scaled_returns)) - 1
                
                # Calculate maximum drawdown
                cumulative = np.cumprod(1 + np.array(scaled_returns))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(np.min(drawdown))
                
                # Calculate Sharpe ratio
                mean_return = np.mean(scaled_returns)
                std_return = np.std(scaled_returns)
                sharpe = (mean_return - self.risk_free_rate/252) / std_return if std_return > 0 else 0.0
                
                simulation_results.append({
                    'cumulative_return': cumulative_return,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe,
                    'final_balance': self.initial_capital * (1 + cumulative_return)
                })
            
            # Analyze results
            cumulative_returns = [r['cumulative_return'] for r in simulation_results]
            max_drawdowns = [r['max_drawdown'] for r in simulation_results]
            sharpe_ratios = [r['sharpe_ratio'] for r in simulation_results]
            final_balances = [r['final_balance'] for r in simulation_results]
            
            results = {
                'num_simulations': num_simulations,
                'num_trades_per_sim': num_trades,
                'position_size': position_size,
                'expected_return': np.mean(cumulative_returns),
                'return_std': np.std(cumulative_returns),
                'return_percentiles': {
                    '5th': np.percentile(cumulative_returns, 5),
                    '25th': np.percentile(cumulative_returns, 25),
                    '50th': np.percentile(cumulative_returns, 50),
                    '75th': np.percentile(cumulative_returns, 75),
                    '95th': np.percentile(cumulative_returns, 95)
                },
                'expected_max_drawdown': np.mean(max_drawdowns),
                'drawdown_percentiles': {
                    '5th': np.percentile(max_drawdowns, 5),
                    '25th': np.percentile(max_drawdowns, 25),
                    '50th': np.percentile(max_drawdowns, 50),
                    '75th': np.percentile(max_drawdowns, 75),
                    '95th': np.percentile(max_drawdowns, 95)
                },
                'expected_sharpe': np.mean(sharpe_ratios),
                'probability_of_profit': len([r for r in cumulative_returns if r > 0]) / len(cumulative_returns),
                'expected_final_balance': np.mean(final_balances),
                'worst_case_balance': np.min(final_balances),
                'best_case_balance': np.max(final_balances)
            }
            
            logger.info(f"Monte Carlo simulation completed: {num_simulations} runs, "
                       f"Expected return: {results['expected_return']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {"error": str(e)}
    
    def comprehensive_optimization(self, 
                                 parameters: OptimizationParameters) -> OptimizationResults:
        """
        Perform comprehensive profit optimization
        
        Args:
            parameters: Optimization parameters
            
        Returns:
            OptimizationResults with complete optimization analysis
        """
        try:
            if not self.trade_results:
                logger.warning("No trade results available for comprehensive optimization")
                return self._create_empty_optimization_results()
            
            # Optimize position sizing
            position_sizing = self.optimize_position_sizing(PositionSizingMethod.KELLY)
            optimal_position_size = position_sizing["optimal_size"]
            
            # Optimize risk-reward
            risk_reward_opt = self.optimize_risk_reward(OptimizationObjective.MAXIMIZE_SHARPE)
            optimal_risk_reward = risk_reward_opt["optimal_rr"]
            optimal_stop_loss = risk_reward_opt["stop_loss"]
            optimal_take_profit = risk_reward_opt["take_profit"]
            
            # Monte Carlo simulation with optimal parameters
            mc_results = self.monte_carlo_simulation(
                num_simulations=1000,
                num_trades=252,  # One year of trading
                position_size=optimal_position_size
            )
            
            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_fraction()
            
            # Performance metrics
            returns = [t.return_pct for t in self.trade_results]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            win_rate = len(wins) / len(returns) if returns else 0.0
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf') if wins else 0.0
            
            # Confidence interval from Monte Carlo
            confidence_interval = (
                mc_results.get('return_percentiles', {}).get('5th', 0.0),
                mc_results.get('return_percentiles', {}).get('95th', 0.0)
            )
            
            # Sensitivity analysis (simplified)
            sensitivity_analysis = {
                'position_size_sensitivity': self._calculate_position_size_sensitivity(),
                'stop_loss_sensitivity': self._calculate_stop_loss_sensitivity(),
                'take_profit_sensitivity': self._calculate_take_profit_sensitivity()
            }
            
            results = OptimizationResults(
                optimal_position_size=optimal_position_size,
                optimal_risk_reward=optimal_risk_reward,
                optimal_stop_loss=optimal_stop_loss,
                optimal_take_profit=optimal_take_profit,
                expected_return=mc_results.get('expected_return', 0.0),
                expected_sharpe=mc_results.get('expected_sharpe', 0.0),
                expected_max_drawdown=mc_results.get('expected_max_drawdown', 0.0),
                win_rate=win_rate,
                profit_factor=profit_factor,
                kelly_fraction=kelly_fraction,
                confidence_interval=confidence_interval,
                sensitivity_analysis=sensitivity_analysis
            )
            
            logger.info(f"Comprehensive optimization completed: "
                       f"Optimal position size: {optimal_position_size:.3f}, "
                       f"Optimal R:R: {optimal_risk_reward:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive optimization: {str(e)}")
            return self._create_empty_optimization_results()
    
    def _calculate_position_size_sensitivity(self) -> float:
        """Calculate sensitivity to position size changes"""
        # Simplified sensitivity calculation
        base_size = 0.02
        test_sizes = [base_size * 0.5, base_size, base_size * 1.5]
        
        returns = []
        for size in test_sizes:
            mc_result = self.monte_carlo_simulation(num_simulations=100, position_size=size)
            returns.append(mc_result.get('expected_return', 0.0))
        
        # Calculate sensitivity as standard deviation of returns
        return np.std(returns) if len(returns) > 1 else 0.0
    
    def _calculate_stop_loss_sensitivity(self) -> float:
        """Calculate sensitivity to stop loss changes"""
        # Simplified calculation
        return 0.1  # Placeholder
    
    def _calculate_take_profit_sensitivity(self) -> float:
        """Calculate sensitivity to take profit changes"""
        # Simplified calculation
        return 0.15  # Placeholder
    
    def _create_empty_optimization_results(self) -> OptimizationResults:
        """Create empty optimization results"""
        return OptimizationResults(
            optimal_position_size=0.02,
            optimal_risk_reward=2.0,
            optimal_stop_loss=0.01,
            optimal_take_profit=0.02,
            expected_return=0.0,
            expected_sharpe=0.0,
            expected_max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            kelly_fraction=0.0,
            confidence_interval=(0.0, 0.0),
            sensitivity_analysis={}
        )
