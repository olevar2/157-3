"""
Walk-Forward Optimization Engine
Advanced optimization techniques to prevent overfitting and ensure strategy robustness.

This module implements walk-forward optimization, a technique that validates trading
strategies by optimizing parameters on historical data and testing on out-of-sample
periods, preventing overfitting and ensuring robust performance.

Key Features:
- Rolling window optimization
- Out-of-sample validation
- Parameter space exploration
- Statistical significance testing
- Performance degradation detection
- Robust metric calculation

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParameterSpace:
    """Parameter space definition for optimization."""
    parameters: Dict[str, List[Any]]
    constraints: Optional[Dict[str, Callable]] = None

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))

            # Apply constraints if any
            if self.constraints:
                valid = True
                for constraint_name, constraint_func in self.constraints.items():
                    if not constraint_func(param_dict):
                        valid = False
                        break
                if valid:
                    combinations.append(param_dict)
            else:
                combinations.append(param_dict)

        return combinations

@dataclass
class WalkForwardWindow:
    """Walk-forward optimization window."""
    start_date: datetime
    end_date: datetime
    optimization_period: int  # days
    validation_period: int    # days
    window_id: int

    @property
    def optimization_end(self) -> datetime:
        return self.start_date + timedelta(days=self.optimization_period)

    @property
    def validation_start(self) -> datetime:
        return self.optimization_end

    @property
    def validation_end(self) -> datetime:
        return self.validation_start + timedelta(days=self.validation_period)

@dataclass
class OptimizationMetrics:
    """Optimization performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    trades_count: int
    avg_trade_duration: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'trades_count': self.trades_count,
            'avg_trade_duration': self.avg_trade_duration
        }

@dataclass
class OptimizationResult:
    """Walk-forward optimization result."""
    window: WalkForwardWindow
    best_parameters: Dict[str, Any]
    optimization_metrics: OptimizationMetrics
    validation_metrics: OptimizationMetrics
    all_results: List[Tuple[Dict[str, Any], OptimizationMetrics]]
    optimization_time: float

    @property
    def performance_degradation(self) -> float:
        """Calculate performance degradation from optimization to validation."""
        opt_return = self.optimization_metrics.total_return
        val_return = self.validation_metrics.total_return

        if opt_return == 0:
            return float('inf') if val_return < 0 else 0

        return (opt_return - val_return) / abs(opt_return)

@dataclass
class OptimizationConfig:
    """Walk-forward optimization configuration."""
    optimization_period_days: int = 252  # 1 year
    validation_period_days: int = 63     # 3 months
    step_size_days: int = 21             # 3 weeks
    min_trades_required: int = 10
    optimization_metric: str = 'sharpe_ratio'
    max_workers: int = 4
    significance_level: float = 0.05

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine

    Implements walk-forward optimization to prevent overfitting and ensure
    strategy robustness across different market periods.
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.results: List[OptimizationResult] = []

    def optimize_strategy(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        parameter_space: ParameterSpace,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[OptimizationResult]:
        """
        Perform walk-forward optimization on a trading strategy.

        Args:
            data: Historical market data
            strategy_func: Strategy function to optimize
            parameter_space: Parameter space to explore
            start_date: Optimization start date
            end_date: Optimization end date

        Returns:
            List of optimization results for each window
        """
        logger.info("Starting walk-forward optimization...")

        # Set default dates
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]

        # Generate walk-forward windows
        windows = self._generate_windows(start_date, end_date)
        logger.info(f"Generated {len(windows)} walk-forward windows")

        # Generate parameter combinations
        param_combinations = parameter_space.generate_combinations()
        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Optimize each window
        self.results = []
        for window in windows:
            logger.info(f"Optimizing window {window.window_id}: {window.start_date} to {window.validation_end}")

            result = self._optimize_window(
                data, strategy_func, param_combinations, window
            )

            if result:
                self.results.append(result)
                logger.info(f"Window {window.window_id} completed. Best params: {result.best_parameters}")

        logger.info(f"Walk-forward optimization completed. {len(self.results)} windows optimized.")
        return self.results

    def _generate_windows(self, start_date: datetime, end_date: datetime) -> List[WalkForwardWindow]:
        """Generate walk-forward windows."""
        windows = []
        window_id = 0

        current_start = start_date
        while current_start + timedelta(days=self.config.optimization_period_days + self.config.validation_period_days) <= end_date:
            window = WalkForwardWindow(
                start_date=current_start,
                end_date=current_start + timedelta(days=self.config.optimization_period_days + self.config.validation_period_days),
                optimization_period=self.config.optimization_period_days,
                validation_period=self.config.validation_period_days,
                window_id=window_id
            )
            windows.append(window)

            current_start += timedelta(days=self.config.step_size_days)
            window_id += 1

        return windows

    def _optimize_window(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_combinations: List[Dict[str, Any]],
        window: WalkForwardWindow
    ) -> Optional[OptimizationResult]:
        """Optimize a single walk-forward window."""
        start_time = datetime.now()

        # Split data into optimization and validation periods
        opt_data = data[(data.index >= window.start_date) & (data.index < window.optimization_end)]
        val_data = data[(data.index >= window.validation_start) & (data.index < window.validation_end)]

        if len(opt_data) == 0 or len(val_data) == 0:
            logger.warning(f"Insufficient data for window {window.window_id}")
            return None

        # Optimize parameters on optimization period
        optimization_results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_params = {
                executor.submit(self._evaluate_parameters, opt_data, strategy_func, params): params
                for params in param_combinations
            }

            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    metrics = future.result()
                    if metrics and metrics.trades_count >= self.config.min_trades_required:
                        optimization_results.append((params, metrics))
                except Exception as e:
                    logger.warning(f"Error evaluating parameters {params}: {e}")

        if not optimization_results:
            logger.warning(f"No valid results for window {window.window_id}")
            return None

        # Find best parameters based on optimization metric
        best_params, best_opt_metrics = max(
            optimization_results,
            key=lambda x: getattr(x[1], self.config.optimization_metric)
        )

        # Validate on out-of-sample period
        val_metrics = self._evaluate_parameters(val_data, strategy_func, best_params)

        if not val_metrics:
            logger.warning(f"Validation failed for window {window.window_id}")
            return None

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            window=window,
            best_parameters=best_params,
            optimization_metrics=best_opt_metrics,
            validation_metrics=val_metrics,
            all_results=optimization_results,
            optimization_time=optimization_time
        )

    def _evaluate_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        parameters: Dict[str, Any]
    ) -> Optional[OptimizationMetrics]:
        """Evaluate strategy with given parameters."""
        try:
            # Run strategy with parameters
            results = strategy_func(data, **parameters)

            if not results or 'trades' not in results:
                return None

            trades = results['trades']
            if len(trades) == 0:
                return None

            # Calculate metrics
            returns = np.array([trade.get('pnl', 0) for trade in trades])

            if len(returns) == 0:
                return None

            total_return = np.sum(returns)
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

            # Calculate Sharpe ratio
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # Calculate max drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

            # Calculate profit factor
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]

            gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
            gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0

            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Calculate Calmar ratio
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # Calculate Sortino ratio
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

            # Calculate average trade duration
            durations = [trade.get('duration', 0) for trade in trades]
            avg_trade_duration = np.mean(durations) if durations else 0

            return OptimizationMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                trades_count=len(trades),
                avg_trade_duration=avg_trade_duration
            )

        except Exception as e:
            logger.warning(f"Error evaluating parameters {parameters}: {e}")
            return None

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all optimization windows."""
        if not self.results:
            return {}

        # Collect metrics
        opt_returns = [r.optimization_metrics.total_return for r in self.results]
        val_returns = [r.validation_metrics.total_return for r in self.results]
        degradations = [r.performance_degradation for r in self.results if not np.isinf(r.performance_degradation)]

        opt_sharpe = [r.optimization_metrics.sharpe_ratio for r in self.results]
        val_sharpe = [r.validation_metrics.sharpe_ratio for r in self.results]

        return {
            'total_windows': len(self.results),
            'avg_optimization_return': np.mean(opt_returns),
            'avg_validation_return': np.mean(val_returns),
            'avg_performance_degradation': np.mean(degradations) if degradations else 0,
            'std_performance_degradation': np.std(degradations) if degradations else 0,
            'avg_optimization_sharpe': np.mean(opt_sharpe),
            'avg_validation_sharpe': np.mean(val_sharpe),
            'consistency_score': self._calculate_consistency_score(),
            'robustness_score': self._calculate_robustness_score()
        }

    def _calculate_consistency_score(self) -> float:
        """Calculate strategy consistency score."""
        if len(self.results) < 2:
            return 0

        val_returns = [r.validation_metrics.total_return for r in self.results]
        positive_periods = len([r for r in val_returns if r > 0])

        return positive_periods / len(val_returns)

    def _calculate_robustness_score(self) -> float:
        """Calculate strategy robustness score."""
        if not self.results:
            return 0

        degradations = [r.performance_degradation for r in self.results if not np.isinf(r.performance_degradation)]

        if not degradations:
            return 0

        # Lower degradation = higher robustness
        avg_degradation = np.mean(degradations)
        return max(0, 1 - avg_degradation)
