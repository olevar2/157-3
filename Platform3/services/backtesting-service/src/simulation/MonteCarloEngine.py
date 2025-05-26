"""
Monte Carlo Simulation Engine for Trading Strategy Analysis

This module provides comprehensive Monte Carlo simulation capabilities for
trading strategy analysis, risk assessment, and performance projection.
It supports multiple simulation methods and provides detailed statistical
analysis of potential outcomes.

Key Features:
- Multiple Monte Carlo simulation methods
- Bootstrap resampling of historical trades
- Path-dependent simulation with market regimes
- Risk metrics and confidence intervals
- Scenario analysis and stress testing
- Portfolio-level simulation capabilities

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimulationType(Enum):
    """Types of Monte Carlo simulations"""
    BOOTSTRAP = "bootstrap"
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    REGIME_SWITCHING = "regime_switching"
    GEOMETRIC_BROWNIAN = "geometric_brownian"

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class Trade:
    """Individual trade data"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    duration_hours: float
    regime: Optional[MarketRegime] = None

@dataclass
class SimulationParameters:
    """Monte Carlo simulation parameters"""
    num_simulations: int = 10000
    simulation_type: SimulationType = SimulationType.BOOTSTRAP
    time_horizon_days: int = 252  # 1 year
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    initial_capital: float = 100000.0
    reinvest_profits: bool = True
    transaction_costs: float = 0.0001  # 0.01%
    max_drawdown_stop: float = 0.20  # 20%
    regime_probabilities: Optional[Dict[MarketRegime, float]] = None

@dataclass
class SimulationResult:
    """Single simulation path result"""
    simulation_id: int
    final_capital: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    daily_returns: np.ndarray
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray

@dataclass
class MonteCarloResults:
    """Comprehensive Monte Carlo analysis results"""
    parameters: SimulationParameters
    num_simulations: int
    execution_time_seconds: float
    
    # Summary statistics
    mean_return: float
    median_return: float
    std_return: float
    skewness: float
    kurtosis: float
    
    # Risk metrics
    var_95: float  # Value at Risk
    var_99: float
    cvar_95: float  # Conditional VaR
    cvar_99: float
    max_drawdown_95: float
    max_drawdown_99: float
    
    # Confidence intervals
    return_confidence_intervals: Dict[float, Tuple[float, float]]
    drawdown_confidence_intervals: Dict[float, Tuple[float, float]]
    
    # Probability metrics
    prob_positive_return: float
    prob_target_return: Dict[float, float]  # Probability of achieving target returns
    prob_max_drawdown: Dict[float, float]   # Probability of exceeding drawdown levels
    
    # Individual simulation results
    simulation_results: List[SimulationResult]
    
    # Performance distribution
    return_distribution: np.ndarray
    drawdown_distribution: np.ndarray

class MonteCarloEngine:
    """
    Advanced Monte Carlo Simulation Engine for Trading Strategy Analysis
    
    Provides comprehensive simulation capabilities for risk assessment,
    performance projection, and scenario analysis of trading strategies.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_multiprocessing: bool = True):
        """
        Initialize Monte Carlo Engine
        
        Args:
            max_workers: Maximum number of worker processes/threads
            use_multiprocessing: Whether to use multiprocessing for parallel execution
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_multiprocessing = use_multiprocessing
        
        # Simulation state
        self.historical_trades = []
        self.market_regimes = {}
        self.regime_transitions = {}
        
        # Performance tracking
        self.simulation_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"✅ MonteCarloEngine initialized with {self.max_workers} workers")

    async def run_simulation(self, 
                           historical_trades: List[Trade],
                           parameters: SimulationParameters) -> MonteCarloResults:
        """
        Run comprehensive Monte Carlo simulation
        
        Args:
            historical_trades: Historical trade data for analysis
            parameters: Simulation parameters
            
        Returns:
            MonteCarloResults with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not historical_trades:
                raise ValueError("No historical trades provided")
            
            self.historical_trades = historical_trades
            
            # Prepare simulation data
            await self._prepare_simulation_data(historical_trades, parameters)
            
            # Run simulations in parallel
            simulation_results = await self._run_parallel_simulations(parameters)
            
            # Analyze results
            monte_carlo_results = self._analyze_simulation_results(
                simulation_results, parameters, start_time
            )
            
            # Update tracking
            self.simulation_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.total_execution_time += execution_time
            
            logger.info(f"✅ Monte Carlo simulation completed: {parameters.num_simulations} runs in {execution_time:.2f}s")
            
            return monte_carlo_results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            raise

    async def _prepare_simulation_data(self, 
                                     trades: List[Trade], 
                                     parameters: SimulationParameters) -> None:
        """Prepare data for simulation"""
        try:
            # Analyze market regimes if using regime-switching simulation
            if parameters.simulation_type == SimulationType.REGIME_SWITCHING:
                self._analyze_market_regimes(trades)
            
            # Calculate trade statistics
            self._calculate_trade_statistics(trades)
            
            # Prepare regime transition probabilities
            if parameters.regime_probabilities:
                self.regime_transitions = self._calculate_regime_transitions(trades)
            
        except Exception as e:
            logger.error(f"Error preparing simulation data: {e}")
            raise

    def _analyze_market_regimes(self, trades: List[Trade]) -> None:
        """Analyze market regimes from historical trades"""
        try:
            # Group trades by time periods
            trade_df = pd.DataFrame([{
                'date': trade.entry_time.date(),
                'pnl_percent': trade.pnl_percent,
                'duration': trade.duration_hours
            } for trade in trades])
            
            # Calculate rolling statistics
            trade_df = trade_df.groupby('date').agg({
                'pnl_percent': ['mean', 'std'],
                'duration': 'mean'
            }).reset_index()
            
            trade_df.columns = ['date', 'avg_return', 'volatility', 'avg_duration']
            
            # Classify regimes based on return and volatility
            vol_threshold = trade_df['volatility'].quantile(0.7)
            return_threshold = 0.0
            
            for _, row in trade_df.iterrows():
                if row['volatility'] > vol_threshold:
                    regime = MarketRegime.HIGH_VOLATILITY
                elif row['avg_return'] > return_threshold:
                    regime = MarketRegime.BULL
                elif row['avg_return'] < -return_threshold:
                    regime = MarketRegime.BEAR
                else:
                    regime = MarketRegime.SIDEWAYS
                
                self.market_regimes[row['date']] = regime
                
        except Exception as e:
            logger.error(f"Error analyzing market regimes: {e}")

    def _calculate_trade_statistics(self, trades: List[Trade]) -> None:
        """Calculate statistical properties of trades"""
        try:
            returns = [trade.pnl_percent for trade in trades]
            
            self.trade_stats = {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'avg_win': np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0,
                'avg_loss': np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0,
                'max_win': max(returns) if returns else 0,
                'max_loss': min(returns) if returns else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            self.trade_stats = {}

    def _calculate_regime_transitions(self, trades: List[Trade]) -> Dict:
        """Calculate regime transition probabilities"""
        try:
            transitions = {}
            regimes = list(self.market_regimes.values())
            
            for i in range(len(regimes) - 1):
                current = regimes[i]
                next_regime = regimes[i + 1]
                
                if current not in transitions:
                    transitions[current] = {}
                
                if next_regime not in transitions[current]:
                    transitions[current][next_regime] = 0
                
                transitions[current][next_regime] += 1
            
            # Normalize to probabilities
            for current in transitions:
                total = sum(transitions[current].values())
                for next_regime in transitions[current]:
                    transitions[current][next_regime] /= total
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error calculating regime transitions: {e}")
            return {}

    async def _run_parallel_simulations(self, parameters: SimulationParameters) -> List[SimulationResult]:
        """Run simulations in parallel"""
        try:
            if self.use_multiprocessing:
                return await self._run_multiprocess_simulations(parameters)
            else:
                return await self._run_threaded_simulations(parameters)
                
        except Exception as e:
            logger.error(f"Error in parallel simulations: {e}")
            return []

    async def _run_multiprocess_simulations(self, parameters: SimulationParameters) -> List[SimulationResult]:
        """Run simulations using multiprocessing"""
        try:
            # Prepare simulation chunks
            chunk_size = max(1, parameters.num_simulations // self.max_workers)
            chunks = []
            
            for i in range(0, parameters.num_simulations, chunk_size):
                end_idx = min(i + chunk_size, parameters.num_simulations)
                chunks.append((i, end_idx, parameters, self.historical_trades, self.trade_stats))
            
            # Run simulations in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(executor, self._run_simulation_chunk, chunk)
                    for chunk in chunks
                ]
                
                chunk_results = await asyncio.gather(*futures)
            
            # Flatten results
            all_results = []
            for chunk_result in chunk_results:
                all_results.extend(chunk_result)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in multiprocess simulations: {e}")
            return []

    async def _run_threaded_simulations(self, parameters: SimulationParameters) -> List[SimulationResult]:
        """Run simulations using threading"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor, 
                        self._run_single_simulation, 
                        i, parameters, self.historical_trades
                    )
                    for i in range(parameters.num_simulations)
                ]
                
                results = await asyncio.gather(*futures)
            
            return [r for r in results if r is not None]
            
        except Exception as e:
            logger.error(f"Error in threaded simulations: {e}")
            return []

    def _run_simulation_chunk(self, chunk_data: Tuple) -> List[SimulationResult]:
        """Run a chunk of simulations (for multiprocessing)"""
        start_idx, end_idx, parameters, historical_trades, trade_stats = chunk_data
        results = []
        
        for i in range(start_idx, end_idx):
            try:
                result = self._run_single_simulation(i, parameters, historical_trades)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in simulation {i}: {e}")
        
        return results

    def _run_single_simulation(self, 
                             simulation_id: int,
                             parameters: SimulationParameters,
                             historical_trades: List[Trade]) -> Optional[SimulationResult]:
        """Run a single Monte Carlo simulation"""
        try:
            # Initialize simulation state
            capital = parameters.initial_capital
            equity_curve = [capital]
            daily_returns = []
            max_drawdown = 0.0
            peak_capital = capital
            
            # Generate trade sequence based on simulation type
            if parameters.simulation_type == SimulationType.BOOTSTRAP:
                simulated_trades = self._bootstrap_trades(historical_trades, parameters)
            elif parameters.simulation_type == SimulationType.PARAMETRIC:
                simulated_trades = self._parametric_trades(parameters)
            elif parameters.simulation_type == SimulationType.GEOMETRIC_BROWNIAN:
                simulated_trades = self._geometric_brownian_trades(parameters)
            else:
                simulated_trades = self._bootstrap_trades(historical_trades, parameters)
            
            # Simulate trading
            winning_trades = 0
            total_trades = len(simulated_trades)
            
            for trade in simulated_trades:
                # Apply transaction costs
                trade_return = trade.pnl_percent - parameters.transaction_costs
                
                # Calculate position size (simplified)
                position_size = capital * 0.02  # 2% risk per trade
                trade_pnl = position_size * trade_return
                
                # Update capital
                if parameters.reinvest_profits:
                    capital += trade_pnl
                else:
                    capital = parameters.initial_capital + (capital - parameters.initial_capital) + trade_pnl
                
                equity_curve.append(capital)
                
                # Track performance
                if trade_return > 0:
                    winning_trades += 1
                
                # Update drawdown
                if capital > peak_capital:
                    peak_capital = capital
                
                current_drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Check drawdown stop
                if current_drawdown > parameters.max_drawdown_stop:
                    break
                
                # Calculate daily return
                if len(equity_curve) > 1:
                    daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                    daily_returns.append(daily_return)
            
            # Calculate final metrics
            total_return = (capital - parameters.initial_capital) / parameters.initial_capital
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate risk-adjusted metrics
            daily_returns_array = np.array(daily_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns_array)
            sortino_ratio = self._calculate_sortino_ratio(daily_returns_array)
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Calculate profit factor
            wins = [r for r in daily_returns if r > 0]
            losses = [r for r in daily_returns if r < 0]
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else 0
            
            # Create drawdown curve
            equity_array = np.array(equity_curve)
            peak_array = np.maximum.accumulate(equity_array)
            drawdown_curve = (peak_array - equity_array) / peak_array
            
            return SimulationResult(
                simulation_id=simulation_id,
                final_capital=capital,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                daily_returns=daily_returns_array,
                equity_curve=equity_array,
                drawdown_curve=drawdown_curve
            )
            
        except Exception as e:
            logger.error(f"Error in simulation {simulation_id}: {e}")
            return None

    def _bootstrap_trades(self, historical_trades: List[Trade], parameters: SimulationParameters) -> List[Trade]:
        """Generate trades using bootstrap resampling"""
        try:
            num_trades = int(parameters.time_horizon_days * len(historical_trades) / 252)  # Scale to time horizon
            return np.random.choice(historical_trades, size=num_trades, replace=True).tolist()
        except Exception:
            return historical_trades[:num_trades] if len(historical_trades) >= num_trades else historical_trades

    def _parametric_trades(self, parameters: SimulationParameters) -> List[Trade]:
        """Generate trades using parametric distribution"""
        try:
            if not hasattr(self, 'trade_stats'):
                return []
            
            num_trades = int(parameters.time_horizon_days * 2)  # Assume 2 trades per day
            
            # Generate returns from fitted distribution
            returns = np.random.normal(
                self.trade_stats['mean_return'],
                self.trade_stats['std_return'],
                num_trades
            )
            
            # Create synthetic trades
            trades = []
            base_time = datetime.now()
            
            for i, ret in enumerate(returns):
                trade = Trade(
                    entry_time=base_time + timedelta(hours=i*12),
                    exit_time=base_time + timedelta(hours=i*12+4),
                    symbol="SYNTHETIC",
                    side="LONG",
                    entry_price=1.0,
                    exit_price=1.0 + ret,
                    quantity=1.0,
                    pnl=ret,
                    pnl_percent=ret,
                    duration_hours=4.0
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error generating parametric trades: {e}")
            return []

    def _geometric_brownian_trades(self, parameters: SimulationParameters) -> List[Trade]:
        """Generate trades using geometric Brownian motion"""
        try:
            if not hasattr(self, 'trade_stats'):
                return []
            
            num_trades = int(parameters.time_horizon_days * 2)
            dt = 1.0 / 252  # Daily time step
            
            mu = self.trade_stats['mean_return']
            sigma = self.trade_stats['std_return']
            
            # Generate price path
            dW = np.random.normal(0, np.sqrt(dt), num_trades)
            price_path = np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dW))
            
            # Convert to trades
            trades = []
            base_time = datetime.now()
            
            for i in range(len(price_path) - 1):
                ret = (price_path[i+1] - price_path[i]) / price_path[i]
                
                trade = Trade(
                    entry_time=base_time + timedelta(hours=i*12),
                    exit_time=base_time + timedelta(hours=i*12+4),
                    symbol="GBM",
                    side="LONG",
                    entry_price=price_path[i],
                    exit_price=price_path[i+1],
                    quantity=1.0,
                    pnl=ret,
                    pnl_percent=ret,
                    duration_hours=4.0
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error generating GBM trades: {e}")
            return []

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            
            excess_returns = np.mean(returns) - risk_free_rate / 252
            return excess_returns / np.std(returns) * np.sqrt(252)
        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = np.mean(returns) - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                return 0.0
            
            return excess_returns / np.std(downside_returns) * np.sqrt(252)
        except Exception:
            return 0.0

    def _analyze_simulation_results(self, 
                                  simulation_results: List[SimulationResult],
                                  parameters: SimulationParameters,
                                  start_time: datetime) -> MonteCarloResults:
        """Analyze and summarize simulation results"""
        try:
            if not simulation_results:
                raise ValueError("No simulation results to analyze")
            
            # Extract return distribution
            returns = [result.total_return for result in simulation_results]
            drawdowns = [result.max_drawdown for result in simulation_results]
            
            returns_array = np.array(returns)
            drawdowns_array = np.array(drawdowns)
            
            # Calculate summary statistics
            mean_return = np.mean(returns_array)
            median_return = np.median(returns_array)
            std_return = np.std(returns_array)
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            cvar_95 = np.mean(returns_array[returns_array <= var_95])
            cvar_99 = np.mean(returns_array[returns_array <= var_99])
            
            # Calculate drawdown statistics
            max_drawdown_95 = np.percentile(drawdowns_array, 95)
            max_drawdown_99 = np.percentile(drawdowns_array, 99)
            
            # Calculate confidence intervals
            return_confidence_intervals = {}
            drawdown_confidence_intervals = {}
            
            for confidence in parameters.confidence_levels:
                alpha = 1 - confidence
                lower_pct = (alpha / 2) * 100
                upper_pct = (1 - alpha / 2) * 100
                
                return_confidence_intervals[confidence] = (
                    np.percentile(returns_array, lower_pct),
                    np.percentile(returns_array, upper_pct)
                )
                
                drawdown_confidence_intervals[confidence] = (
                    np.percentile(drawdowns_array, lower_pct),
                    np.percentile(drawdowns_array, upper_pct)
                )
            
            # Calculate probability metrics
            prob_positive_return = np.mean(returns_array > 0)
            
            # Target return probabilities
            target_returns = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
            prob_target_return = {
                target: np.mean(returns_array >= target) 
                for target in target_returns
            }
            
            # Drawdown probabilities
            drawdown_levels = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
            prob_max_drawdown = {
                level: np.mean(drawdowns_array >= level)
                for level in drawdown_levels
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return MonteCarloResults(
                parameters=parameters,
                num_simulations=len(simulation_results),
                execution_time_seconds=execution_time,
                mean_return=mean_return,
                median_return=median_return,
                std_return=std_return,
                skewness=skewness,
                kurtosis=kurtosis,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown_95=max_drawdown_95,
                max_drawdown_99=max_drawdown_99,
                return_confidence_intervals=return_confidence_intervals,
                drawdown_confidence_intervals=drawdown_confidence_intervals,
                prob_positive_return=prob_positive_return,
                prob_target_return=prob_target_return,
                prob_max_drawdown=prob_max_drawdown,
                simulation_results=simulation_results,
                return_distribution=returns_array,
                drawdown_distribution=drawdowns_array
            )
            
        except Exception as e:
            logger.error(f"Error analyzing simulation results: {e}")
            raise

    def get_performance_stats(self) -> Dict:
        """Get engine performance statistics"""
        return {
            'simulation_count': self.simulation_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.total_execution_time / max(1, self.simulation_count),
            'max_workers': self.max_workers,
            'use_multiprocessing': self.use_multiprocessing
        }
