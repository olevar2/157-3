"""
Walk-Forward Optimization System for Trading Strategies

This module provides comprehensive walk-forward optimization with advanced features
for parameter optimization, regime detection, and robust backtesting methodologies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
from scipy import stats
import warnings
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class OptimizationParameters:
    """Parameter space definition for optimization"""
    parameter_grid: Dict[str, List[Any]]
    constraints: Optional[Dict[str, Callable]] = None
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
    parameter_types: Optional[Dict[str, str]] = None  # 'continuous', 'discrete', 'categorical'


@dataclass
class WalkForwardWindow:
    """Walk-forward optimization window configuration"""
    in_sample_periods: int
    out_sample_periods: int
    step_size: int
    minimum_sample_size: int
    reoptimization_frequency: int


@dataclass
class OptimizationResult:
    """Results from walk-forward optimization"""
    best_parameters: Dict[str, Any]
    parameter_stability: float
    in_sample_performance: Dict[str, float]
    out_sample_performance: Dict[str, float]
    parameter_evolution: List[Dict[str, Any]]
    window_results: List[Dict[str, Any]]
    optimization_statistics: Dict[str, float]
    regime_analysis: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    overfitting_score: float
    recommendation: str


@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    var_95: float
    information_ratio: float


class WalkForwardOptimizer:
    """
    Advanced Walk-Forward Optimization System
    
    Features:
    - Multi-objective optimization
    - Parameter stability analysis
    - Regime-aware optimization
    - Robust statistical testing
    - Parallel processing
    - Overfitting detection
    """
    
    def __init__(self,
                 optimization_metric: str = 'sharpe_ratio',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize WalkForwardOptimizer
        
        Args:
            optimization_metric: Primary metric for optimization
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        
        # Supported metrics
        self.supported_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return',
            'max_drawdown', 'win_rate', 'profit_factor', 'information_ratio'
        ]
        
        # Performance thresholds
        self.performance_thresholds = {
            'sharpe_ratio': 1.0,
            'sortino_ratio': 1.2,
            'calmar_ratio': 0.5,
            'max_drawdown': -0.15,
            'win_rate': 0.5,
            'profit_factor': 1.2
        }
        
        np.random.seed(random_state)
        self.logger.info("WalkForwardOptimizer initialized with advanced optimization features")

    def optimize(self, 
                data: pd.DataFrame,
                strategy_function: Callable,
                parameters: OptimizationParameters,
                window_config: WalkForwardWindow,
                multi_objective: bool = False,
                regime_analysis: bool = True) -> OptimizationResult:
        """
        Run walk-forward optimization
        
        Args:
            data: Historical market data
            strategy_function: Strategy function to optimize
            parameters: Parameter space definition
            window_config: Walk-forward window configuration
            multi_objective: Whether to use multi-objective optimization
            regime_analysis: Whether to perform regime analysis
            
        Returns:
            OptimizationResult: Comprehensive optimization results
        """
        try:
            self.logger.info("Starting walk-forward optimization")
            
            # Validate inputs
            self._validate_inputs(data, parameters, window_config)
            
            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(data, window_config)
            self.logger.info(f"Generated {len(windows)} walk-forward windows")
            
            # Perform optimization across all windows
            window_results = []
            parameter_evolution = []
            
            for i, window in enumerate(windows):
                self.logger.info(f"Optimizing window {i+1}/{len(windows)}")
                
                # Extract window data
                in_sample_data = data.iloc[window['in_sample_start']:window['in_sample_end']]
                out_sample_data = data.iloc[window['out_sample_start']:window['out_sample_end']]
                
                # Optimize parameters for this window
                if multi_objective:
                    best_params, window_performance = self._multi_objective_optimization(
                        in_sample_data, strategy_function, parameters
                    )
                else:
                    best_params, window_performance = self._single_objective_optimization(
                        in_sample_data, strategy_function, parameters
                    )
                
                # Evaluate on out-of-sample data
                oos_performance = self._evaluate_strategy(
                    out_sample_data, strategy_function, best_params
                )
                
                # Store results
                window_result = {
                    'window_id': i,
                    'in_sample_start': window['in_sample_start'],
                    'in_sample_end': window['in_sample_end'],
                    'out_sample_start': window['out_sample_start'],
                    'out_sample_end': window['out_sample_end'],
                    'best_parameters': best_params,
                    'in_sample_performance': window_performance,
                    'out_sample_performance': oos_performance
                }
                
                window_results.append(window_result)
                parameter_evolution.append(best_params)
            
            # Analyze optimization results
            analysis_results = self._analyze_optimization_results(
                window_results, parameter_evolution, regime_analysis, data
            )
            
            # Create comprehensive result object
            optimization_result = OptimizationResult(
                best_parameters=analysis_results['consensus_parameters'],
                parameter_stability=analysis_results['parameter_stability'],
                in_sample_performance=analysis_results['aggregate_in_sample'],
                out_sample_performance=analysis_results['aggregate_out_sample'],
                parameter_evolution=parameter_evolution,
                window_results=window_results,
                optimization_statistics=analysis_results['optimization_stats'],
                regime_analysis=analysis_results.get('regime_analysis', {}),
                confidence_intervals=analysis_results['confidence_intervals'],
                overfitting_score=analysis_results['overfitting_score'],
                recommendation=analysis_results['recommendation']
            )
            
            # Store in history
            self.optimization_history.append(optimization_result)
            
            self.logger.info("Walk-forward optimization completed successfully")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward optimization: {str(e)}")
            raise

    def _generate_walk_forward_windows(self, 
                                     data: pd.DataFrame, 
                                     config: WalkForwardWindow) -> List[Dict[str, int]]:
        """Generate walk-forward optimization windows"""
        windows = []
        data_length = len(data)
        
        current_start = 0
        
        while current_start + config.in_sample_periods + config.out_sample_periods <= data_length:
            in_sample_start = current_start
            in_sample_end = current_start + config.in_sample_periods
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + config.out_sample_periods
            
            # Ensure minimum sample size
            if in_sample_end - in_sample_start >= config.minimum_sample_size:
                windows.append({
                    'in_sample_start': in_sample_start,
                    'in_sample_end': in_sample_end,
                    'out_sample_start': out_sample_start,
                    'out_sample_end': out_sample_end
                })
            
            current_start += config.step_size
            
        return windows

    def _single_objective_optimization(self,
                                     data: pd.DataFrame,
                                     strategy_function: Callable,
                                     parameters: OptimizationParameters) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform single-objective optimization"""
        try:
            # Generate parameter combinations
            param_grid = ParameterGrid(parameters.parameter_grid)
            
            # Apply constraints if provided
            if parameters.constraints:
                param_grid = [params for params in param_grid 
                             if self._check_constraints(params, parameters.constraints)]
            
            best_score = float('-inf')
            best_params = None
            best_performance = None
            
            # Parallel evaluation of parameter combinations
            if self.n_jobs == 1:
                # Sequential processing
                for params in param_grid:
                    performance = self._evaluate_strategy(data, strategy_function, params)
                    score = performance.get(self.optimization_metric, float('-inf'))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_performance = performance
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                    future_to_params = {
                        executor.submit(self._evaluate_strategy, data, strategy_function, params): params
                        for params in param_grid
                    }
                    
                    for future in as_completed(future_to_params):
                        params = future_to_params[future]
                        try:
                            performance = future.result()
                            score = performance.get(self.optimization_metric, float('-inf'))
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_performance = performance
                        except Exception as e:
                            self.logger.warning(f"Error evaluating parameters {params}: {str(e)}")
            
            return best_params or {}, best_performance or {}
            
        except Exception as e:
            self.logger.error(f"Error in single objective optimization: {str(e)}")
            return {}, {}

    def _multi_objective_optimization(self,
                                    data: pd.DataFrame,
                                    strategy_function: Callable,
                                    parameters: OptimizationParameters) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform multi-objective optimization using weighted scoring"""
        try:
            # Define objective weights
            objective_weights = {
                'sharpe_ratio': 0.3,
                'sortino_ratio': 0.2,
                'calmar_ratio': 0.2,
                'max_drawdown': 0.15,  # Minimize (negative weight applied)
                'win_rate': 0.15
            }
            
            param_grid = ParameterGrid(parameters.parameter_grid)
            
            if parameters.constraints:
                param_grid = [params for params in param_grid 
                             if self._check_constraints(params, parameters.constraints)]
            
            best_score = float('-inf')
            best_params = None
            best_performance = None
            
            for params in param_grid:
                performance = self._evaluate_strategy(data, strategy_function, params)
                
                # Calculate weighted multi-objective score
                score = 0.0
                for metric, weight in objective_weights.items():
                    metric_value = performance.get(metric, 0.0)
                    
                    # Normalize and apply weight
                    if metric == 'max_drawdown':
                        # For drawdown, better is less negative (closer to 0)
                        normalized_value = max(0, 1 + metric_value)  # Convert to positive
                    else:
                        normalized_value = max(0, metric_value)
                    
                    score += weight * normalized_value
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_performance = performance
            
            return best_params or {}, best_performance or {}
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {str(e)}")
            return {}, {}

    def _evaluate_strategy(self, 
                          data: pd.DataFrame, 
                          strategy_function: Callable, 
                          parameters: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy performance with given parameters"""
        try:
            # Run strategy
            strategy_results = strategy_function(data, **parameters)
            
            # Extract returns
            if isinstance(strategy_results, dict):
                returns = strategy_results.get('returns', [])
            elif isinstance(strategy_results, (list, np.ndarray)):
                returns = strategy_results
            else:
                returns = []
            
            if not returns or len(returns) < 2:
                return self._get_default_performance()
            
            # Calculate comprehensive performance metrics
            performance = self._calculate_performance_metrics(returns)
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Error evaluating strategy: {str(e)}")
            return self._get_default_performance()

    def _calculate_performance_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            returns_array = np.array(returns)
            
            if len(returns_array) < 2:
                return self._get_default_performance()
            
            # Basic metrics
            total_return = np.prod(1 + returns_array) - 1
            
            # Risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
            sortino_ratio = self._calculate_sortino_ratio(returns_array)
            calmar_ratio = self._calculate_calmar_ratio(returns_array)
            
            # Drawdown analysis
            max_drawdown = self._calculate_max_drawdown(returns_array)
            
            # Win rate
            win_rate = np.sum(returns_array > 0) / len(returns_array)
            
            # Profit factor
            gross_profit = np.sum(returns_array[returns_array > 0])
            gross_loss = abs(np.sum(returns_array[returns_array < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns_array, 5)
            
            # Information ratio (assuming benchmark return of 0)
            information_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'calmar_ratio': calmar_ratio,
                'var_95': var_95,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {str(e)}")
            return self._get_default_performance()

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252
            if np.std(excess_returns) == 0:
                return 0.0
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        except:
            return 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                return 0.0
            return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        except:
            return 0.0

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        try:
            annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
            max_dd = abs(self._calculate_max_drawdown(returns))
            return annual_return / max_dd if max_dd > 0 else 0.0
        except:
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except:
            return 0.0

    def _analyze_optimization_results(self,
                                    window_results: List[Dict[str, Any]],
                                    parameter_evolution: List[Dict[str, Any]],
                                    regime_analysis: bool,
                                    data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimization results across all windows"""
        try:
            # Calculate parameter stability
            parameter_stability = self._calculate_parameter_stability(parameter_evolution)
            
            # Aggregate performance metrics
            in_sample_metrics = []
            out_sample_metrics = []
            
            for result in window_results:
                in_sample_metrics.append(result['in_sample_performance'])
                out_sample_metrics.append(result['out_sample_performance'])
            
            aggregate_in_sample = self._aggregate_performance_metrics(in_sample_metrics)
            aggregate_out_sample = self._aggregate_performance_metrics(out_sample_metrics)
            
            # Determine consensus parameters
            consensus_parameters = self._determine_consensus_parameters(parameter_evolution)
            
            # Calculate optimization statistics
            optimization_stats = self._calculate_optimization_statistics(
                aggregate_in_sample, aggregate_out_sample
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(out_sample_metrics)
            
            # Calculate overfitting score
            overfitting_score = self._calculate_overfitting_score(
                aggregate_in_sample, aggregate_out_sample
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                parameter_stability, overfitting_score, aggregate_out_sample
            )
            
            results = {
                'parameter_stability': parameter_stability,
                'aggregate_in_sample': aggregate_in_sample,
                'aggregate_out_sample': aggregate_out_sample,
                'consensus_parameters': consensus_parameters,
                'optimization_stats': optimization_stats,
                'confidence_intervals': confidence_intervals,
                'overfitting_score': overfitting_score,
                'recommendation': recommendation
            }
            
            # Add regime analysis if requested
            if regime_analysis:
                results['regime_analysis'] = self._perform_regime_analysis(
                    window_results, data
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing optimization results: {str(e)}")
            return {}

    def _calculate_parameter_stability(self, parameter_evolution: List[Dict[str, Any]]) -> float:
        """Calculate parameter stability across windows"""
        try:
            if len(parameter_evolution) < 2:
                return 1.0
            
            # Calculate coefficient of variation for each parameter
            param_stabilities = []
            
            # Get all parameter names
            all_params = set()
            for params in parameter_evolution:
                all_params.update(params.keys())
            
            for param_name in all_params:
                param_values = [params.get(param_name, 0) for params in parameter_evolution]
                
                # Skip non-numeric parameters
                try:
                    param_values = [float(val) for val in param_values]
                except:
                    continue
                
                if len(param_values) > 1 and np.std(param_values) > 0:
                    cv = np.std(param_values) / (abs(np.mean(param_values)) + 1e-8)
                    stability = 1.0 / (1.0 + cv)
                    param_stabilities.append(stability)
            
            return np.mean(param_stabilities) if param_stabilities else 1.0
            
        except Exception:
            return 0.5

    def _aggregate_performance_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate performance metrics across windows"""
        try:
            if not metrics_list:
                return self._get_default_performance()
            
            aggregated = {}
            
            # Get all metric names
            all_metrics = set()
            for metrics in metrics_list:
                all_metrics.update(metrics.keys())
            
            for metric in all_metrics:
                values = [metrics.get(metric, 0.0) for metrics in metrics_list]
                aggregated[metric] = np.mean(values)
            
            return aggregated
            
        except Exception:
            return self._get_default_performance()

    def _determine_consensus_parameters(self, parameter_evolution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine consensus parameters across all windows"""
        try:
            if not parameter_evolution:
                return {}
            
            consensus = {}
            
            # Get all parameter names
            all_params = set()
            for params in parameter_evolution:
                all_params.update(params.keys())
            
            for param_name in all_params:
                param_values = [params.get(param_name) for params in parameter_evolution if param_name in params]
                
                if not param_values:
                    continue
                
                # For numeric parameters, use median
                try:
                    numeric_values = [float(val) for val in param_values]
                    consensus[param_name] = np.median(numeric_values)
                except:
                    # For categorical parameters, use mode
                    from collections import Counter
                    counter = Counter(param_values)
                    consensus[param_name] = counter.most_common(1)[0][0]
            
            return consensus
            
        except Exception:
            return {}

    def _calculate_optimization_statistics(self,
                                         in_sample: Dict[str, float],
                                         out_sample: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimization statistics"""
        try:
            stats = {}
            
            # Performance degradation
            for metric in ['sharpe_ratio', 'sortino_ratio', 'total_return']:
                in_val = in_sample.get(metric, 0.0)
                out_val = out_sample.get(metric, 0.0)
                
                if in_val != 0:
                    degradation = (in_val - out_val) / abs(in_val)
                    stats[f'{metric}_degradation'] = max(0.0, degradation)
                else:
                    stats[f'{metric}_degradation'] = 0.0
            
            return stats
            
        except Exception:
            return {}

    def _calculate_confidence_intervals(self, 
                                      out_sample_metrics: List[Dict[str, float]],
                                      confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for out-of-sample metrics"""
        try:
            confidence_intervals = {}
            
            # Get all metric names
            all_metrics = set()
            for metrics in out_sample_metrics:
                all_metrics.update(metrics.keys())
            
            for metric in all_metrics:
                values = [metrics.get(metric, 0.0) for metrics in out_sample_metrics]
                
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # Calculate confidence interval
                    alpha = 1 - confidence
                    t_val = stats.t.ppf(1 - alpha/2, len(values) - 1)
                    margin = t_val * std_val / np.sqrt(len(values))
                    
                    confidence_intervals[metric] = (mean_val - margin, mean_val + margin)
                else:
                    confidence_intervals[metric] = (values[0], values[0]) if values else (0.0, 0.0)
            
            return confidence_intervals
            
        except Exception:
            return {}

    def _calculate_overfitting_score(self,
                                   in_sample: Dict[str, float],
                                   out_sample: Dict[str, float]) -> float:
        """Calculate overfitting score"""
        try:
            # Compare key metrics
            score = 0.0
            weight_sum = 0.0
            
            metric_weights = {
                'sharpe_ratio': 0.4,
                'sortino_ratio': 0.3,
                'total_return': 0.3
            }
            
            for metric, weight in metric_weights.items():
                in_val = in_sample.get(metric, 0.0)
                out_val = out_sample.get(metric, 0.0)
                
                if in_val > 0:
                    degradation = max(0.0, (in_val - out_val) / in_val)
                    score += weight * degradation
                    weight_sum += weight
            
            return score / weight_sum if weight_sum > 0 else 0.0
            
        except Exception:
            return 0.5

    def _generate_recommendation(self,
                               parameter_stability: float,
                               overfitting_score: float,
                               out_sample_performance: Dict[str, float]) -> str:
        """Generate optimization recommendation"""
        try:
            sharpe = out_sample_performance.get('sharpe_ratio', 0.0)
            
            if overfitting_score > 0.3:
                return "High overfitting detected. Consider simpler models or more data."
            elif parameter_stability < 0.5:
                return "Low parameter stability. Consider wider parameter ranges or longer windows."
            elif sharpe < 0.5:
                return "Poor out-of-sample performance. Review strategy logic."
            elif sharpe > 1.0 and parameter_stability > 0.7:
                return "Excellent optimization results. Strategy ready for live trading."
            else:
                return "Moderate optimization results. Consider further refinement."
                
        except Exception:
            return "Unable to generate recommendation due to analysis errors."

    def _perform_regime_analysis(self, 
                               window_results: List[Dict[str, Any]], 
                               data: pd.DataFrame) -> Dict[str, Any]:
        """Perform regime analysis on optimization results"""
        try:
            # Simple regime detection using volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=30).std()
            
            high_vol_threshold = volatility.quantile(0.7)
            low_vol_threshold = volatility.quantile(0.3)
            
            regime_performance = {'high_volatility': [], 'low_volatility': [], 'normal': []}
            
            for result in window_results:
                window_start = result['out_sample_start']
                window_end = result['out_sample_end']
                
                window_volatility = volatility.iloc[window_start:window_end].mean()
                performance = result['out_sample_performance']
                
                if window_volatility > high_vol_threshold:
                    regime_performance['high_volatility'].append(performance)
                elif window_volatility < low_vol_threshold:
                    regime_performance['low_volatility'].append(performance)
                else:
                    regime_performance['normal'].append(performance)
            
            # Aggregate regime performance
            regime_analysis = {}
            for regime, performances in regime_performance.items():
                if performances:
                    regime_analysis[regime] = self._aggregate_performance_metrics(performances)
                else:
                    regime_analysis[regime] = self._get_default_performance()
            
            return regime_analysis
            
        except Exception:
            return {}

    def _check_constraints(self, parameters: Dict[str, Any], constraints: Dict[str, Callable]) -> bool:
        """Check if parameters satisfy constraints"""
        try:
            for constraint_name, constraint_func in constraints.items():
                if not constraint_func(parameters):
                    return False
            return True
        except:
            return True

    def _get_default_performance(self) -> Dict[str, float]:
        """Get default performance metrics"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.5,
            'profit_factor': 1.0,
            'calmar_ratio': 0.0,
            'var_95': 0.0,
            'information_ratio': 0.0
        }

    def _validate_inputs(self,
                        data: pd.DataFrame,
                        parameters: OptimizationParameters,
                        window_config: WalkForwardWindow):
        """Validate optimization inputs"""
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if not parameters.parameter_grid:
            raise ValueError("Parameter grid cannot be empty")
        
        if window_config.in_sample_periods <= 0:
            raise ValueError("In-sample periods must be positive")
        
        if window_config.out_sample_periods <= 0:
            raise ValueError("Out-of-sample periods must be positive")

    def generate_optimization_report(self,
                                   result: OptimizationResult,
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "optimization_summary": {
                    "best_parameters": result.best_parameters,
                    "parameter_stability": result.parameter_stability,
                    "overfitting_score": result.overfitting_score,
                    "recommendation": result.recommendation
                },
                "performance_summary": {
                    "in_sample": result.in_sample_performance,
                    "out_sample": result.out_sample_performance,
                    "confidence_intervals": result.confidence_intervals
                },
                "detailed_analysis": {
                    "parameter_evolution": result.parameter_evolution,
                    "optimization_statistics": result.optimization_statistics,
                    "regime_analysis": result.regime_analysis
                },
                "window_results": result.window_results
            }
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return [asdict(result) for result in self.optimization_history]

    def clear_history(self):
        """Clear optimization history"""
        self.optimization_history.clear()
        self.logger.info("Optimization history cleared")


# Example usage and testing
if __name__ == "__main__":
    # Sample strategy function
    def sample_strategy(data, short_window=10, long_window=30, threshold=0.01):
        """Sample moving average crossover strategy"""
        try:
            if 'close' not in data.columns:
                return {'returns': []}
            
            short_ma = data['close'].rolling(window=short_window).mean()
            long_ma = data['close'].rolling(window=long_window).mean()
            
            signals = (short_ma > long_ma * (1 + threshold)).astype(int)
            positions = signals.diff().fillna(0)
            
            returns = (positions.shift(1) * data['close'].pct_change()).dropna()
            
            return {'returns': returns.tolist()}
            
        except Exception:
            return {'returns': []}
    
    # Sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # Optimization parameters
    parameters = OptimizationParameters(
        parameter_grid={
            'short_window': [5, 10, 15, 20],
            'long_window': [20, 30, 40, 50],
            'threshold': [0.005, 0.01, 0.015, 0.02]
        },
        constraints={
            'window_constraint': lambda params: params['long_window'] > params['short_window']
        }
    )
    
    # Window configuration
    window_config = WalkForwardWindow(
        in_sample_periods=252,  # 1 year
        out_sample_periods=63,  # 3 months
        step_size=63,          # 3 months
        minimum_sample_size=100,
        reoptimization_frequency=4
    )
    
    # Initialize optimizer
    optimizer = WalkForwardOptimizer(
        optimization_metric='sharpe_ratio',
        n_jobs=1,  # Use 1 for this example
        verbose=True
    )
    
    # Run optimization
    result = optimizer.optimize(
        data=sample_data,
        strategy_function=sample_strategy,
        parameters=parameters,
        window_config=window_config,
        multi_objective=False,
        regime_analysis=True
    )
    
    print(f"Best Parameters: {result.best_parameters}")
    print(f"Parameter Stability: {result.parameter_stability:.3f}")
    print(f"Out-of-Sample Sharpe: {result.out_sample_performance.get('sharpe_ratio', 0):.3f}")
    print(f"Overfitting Score: {result.overfitting_score:.3f}")
    print(f"Recommendation: {result.recommendation}")
