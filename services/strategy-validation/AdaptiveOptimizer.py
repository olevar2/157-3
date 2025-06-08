#!/usr/bin/env python3
"""
Advanced Adaptive Strategy Optimization System for Platform3

This module provides comprehensive adaptive optimization capabilities including:
- Self-adapting parameter optimization based on market conditions
- Multi-objective optimization balancing return, risk, and stability
- Real-time parameter adjustment based on performance feedback
- Machine learning-driven optimization with regime detection
- Ensemble optimization combining multiple optimization algorithms
- Dynamic risk management integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, differential_evolution, dual_annealing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import optuna
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationTarget:
    """Container for optimization targets and constraints"""
    name: str
    weight: float
    target_type: str  # 'maximize', 'minimize', 'target'
    target_value: Optional[float] = None
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None

@dataclass
class ParameterConfig:
    """Configuration for a single parameter"""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    current_value: Any = None
    importance_score: float = 0.0

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    strategy_id: str
    optimization_method: str
    optimized_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    improvement_factor: float
    confidence_score: float
    market_regime: str
    optimization_duration: float
    iteration_count: int
    convergence_achieved: bool

class AdaptiveOptimizer:
    """
    Advanced adaptive strategy optimization system with machine learning
    """
    
    def __init__(self,
                 optimization_window: int = 252,  # Trading days for optimization
                 reoptimization_frequency: int = 21,  # Days between reoptimization
                 min_trades_threshold: int = 50,
                 max_iterations: int = 1000,
                 convergence_tolerance: float = 1e-6):
        """
        Initialize the Adaptive Optimizer
        
        Args:
            optimization_window: Number of trading days to use for optimization
            reoptimization_frequency: How often to trigger reoptimization
            min_trades_threshold: Minimum trades required for optimization
            max_iterations: Maximum optimization iterations
            convergence_tolerance: Convergence tolerance for optimization
        """
        self.optimization_window = optimization_window
        self.reoptimization_frequency = reoptimization_frequency
        self.min_trades_threshold = min_trades_threshold
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        # Optimization state
        self.strategies = {}
        self.parameter_configs = {}
        self.optimization_targets = {}
        self.optimization_history = {}
        self.market_regimes = {}
        self.performance_predictors = {}
        
        # Machine learning components
        self.regime_detector = None
        self.parameter_importance_model = None
        self.performance_forecaster = None
        
        logger.info("AdaptiveOptimizer initialized with advanced ML capabilities")
    
    def register_strategy(self,
                         strategy_id: str,
                         parameter_configs: List[ParameterConfig],
                         optimization_targets: List[OptimizationTarget],
                         evaluation_function: Callable) -> None:
        """
        Register a strategy for adaptive optimization
        
        Args:
            strategy_id: Unique identifier for the strategy
            parameter_configs: List of parameter configurations
            optimization_targets: List of optimization objectives
            evaluation_function: Function to evaluate strategy performance
        """
        try:
            self.strategies[strategy_id] = {
                'evaluation_function': evaluation_function,
                'last_optimization': None,
                'performance_history': [],
                'parameter_history': [],
                'current_parameters': {param.name: param.current_value for param in parameter_configs}
            }
            
            self.parameter_configs[strategy_id] = {param.name: param for param in parameter_configs}
            self.optimization_targets[strategy_id] = {target.name: target for target in optimization_targets}
            self.optimization_history[strategy_id] = []
            
            logger.info(f"Registered strategy {strategy_id} with {len(parameter_configs)} parameters and {len(optimization_targets)} targets")
            
        except Exception as e:
            logger.error(f"Error registering strategy {strategy_id}: {str(e)}")
            raise
    
    def optimize_strategy(self,
                         strategy_id: str,
                         market_data: pd.DataFrame,
                         method: str = 'adaptive_ensemble',
                         parallel_execution: bool = True) -> OptimizationResult:
        """
        Perform adaptive optimization for a strategy
        
        Args:
            strategy_id: Strategy identifier
            market_data: Historical market data for optimization
            method: Optimization method ('bayesian', 'genetic', 'simulated_annealing', 'adaptive_ensemble')
            parallel_execution: Whether to use parallel processing
            
        Returns:
            OptimizationResult with optimized parameters and metrics
        """
        try:
            if strategy_id not in self.strategies:
                raise ValueError(f"Strategy {strategy_id} not registered")
            
            start_time = datetime.now()
            
            # Detect current market regime
            current_regime = self._detect_market_regime(market_data)
            
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(strategy_id, market_data)
            
            # Select optimization method based on market regime and strategy characteristics
            if method == 'adaptive_ensemble':
                method = self._select_optimal_method(strategy_id, current_regime, optimization_data)
            
            # Perform optimization
            if method == 'bayesian':
                result = self._bayesian_optimization(strategy_id, optimization_data, parallel_execution)
            elif method == 'genetic':
                result = self._genetic_optimization(strategy_id, optimization_data)
            elif method == 'simulated_annealing':
                result = self._simulated_annealing_optimization(strategy_id, optimization_data)
            elif method == 'ensemble':
                result = self._ensemble_optimization(strategy_id, optimization_data, parallel_execution)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate optimization metrics
            optimization_duration = (datetime.now() - start_time).total_seconds()
            
            # Evaluate improvement
            current_performance = self._evaluate_current_performance(strategy_id, optimization_data)
            optimized_performance = self._evaluate_optimized_performance(strategy_id, result['parameters'], optimization_data)
            
            improvement_factor = self._calculate_improvement_factor(current_performance, optimized_performance)
            confidence_score = self._calculate_confidence_score(result, optimization_data)
            
            # Create optimization result
            optimization_result = OptimizationResult(
                strategy_id=strategy_id,
                optimization_method=method,
                optimized_parameters=result['parameters'],
                performance_metrics=optimized_performance,
                improvement_factor=improvement_factor,
                confidence_score=confidence_score,
                market_regime=current_regime,
                optimization_duration=optimization_duration,
                iteration_count=result.get('iterations', 0),
                convergence_achieved=result.get('converged', False)
            )
            
            # Update strategy state
            self._update_strategy_state(strategy_id, optimization_result)
            
            # Learn from optimization
            self._update_learning_models(strategy_id, optimization_result, optimization_data)
            
            logger.info(f"Optimization completed for {strategy_id}: {improvement_factor:.2%} improvement")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing strategy {strategy_id}: {str(e)}")
            raise
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime using clustering and technical indicators"""
        try:
            # Calculate regime indicators
            returns = market_data['close'].pct_change().dropna()
            
            # Volatility regime
            rolling_vol = returns.rolling(20).std()
            current_vol = rolling_vol.iloc[-1]
            vol_percentile = (rolling_vol <= current_vol).mean()
            
            # Trend regime
            sma_short = market_data['close'].rolling(10).mean()
            sma_long = market_data['close'].rolling(50).mean()
            trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            # Volume regime
            if 'volume' in market_data.columns:
                avg_volume = market_data['volume'].rolling(20).mean()
                current_volume_ratio = market_data['volume'].iloc[-1] / avg_volume.iloc[-1]
            else:
                current_volume_ratio = 1.0
            
            # Simple regime classification
            if vol_percentile > 0.8:
                regime = 'high_volatility'
            elif vol_percentile < 0.2:
                regime = 'low_volatility'
            elif trend_strength > 0.05:
                regime = 'bullish_trend'
            elif trend_strength < -0.05:
                regime = 'bearish_trend'
            else:
                regime = 'sideways'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'
    
    def _prepare_optimization_data(self, strategy_id: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for optimization process"""
        try:
            # Limit data to optimization window
            optimization_data = market_data.tail(self.optimization_window).copy()
            
            # Add technical indicators for regime detection
            optimization_data['returns'] = optimization_data['close'].pct_change()
            optimization_data['volatility'] = optimization_data['returns'].rolling(20).std()
            optimization_data['rsi'] = self._calculate_rsi(optimization_data['close'])
            optimization_data['macd'] = self._calculate_macd(optimization_data['close'])
            
            return {
                'market_data': optimization_data,
                'strategy_id': strategy_id,
                'parameter_bounds': self._get_parameter_bounds(strategy_id),
                'optimization_targets': self.optimization_targets[strategy_id],
                'evaluation_function': self.strategies[strategy_id]['evaluation_function']
            }
            
        except Exception as e:
            logger.error(f"Error preparing optimization data: {str(e)}")
            raise
    
    def _bayesian_optimization(self, strategy_id: str, optimization_data: Dict[str, Any], parallel: bool = True) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna"""
        try:
            def objective(trial):
                # Sample parameters
                parameters = {}
                for param_name, param_config in self.parameter_configs[strategy_id].items():
                    if param_config.param_type == 'continuous':
                        parameters[param_name] = trial.suggest_float(
                            param_name, param_config.min_value, param_config.max_value
                        )
                    elif param_config.param_type == 'discrete':
                        parameters[param_name] = trial.suggest_int(
                            param_name, int(param_config.min_value), int(param_config.max_value)
                        )
                    elif param_config.param_type == 'categorical':
                        parameters[param_name] = trial.suggest_categorical(
                            param_name, param_config.choices
                        )
                
                # Evaluate parameters
                try:
                    performance = optimization_data['evaluation_function'](parameters, optimization_data['market_data'])
                    
                    # Multi-objective optimization
                    weighted_score = 0
                    total_weight = 0
                    
                    for target_name, target in optimization_data['optimization_targets'].items():
                        if target_name in performance:
                            value = performance[target_name]
                            
                            if target.target_type == 'maximize':
                                score = value
                            elif target.target_type == 'minimize':
                                score = -value
                            else:  # target
                                score = -abs(value - target.target_value)
                            
                            weighted_score += score * target.weight
                            total_weight += target.weight
                    
                    return weighted_score / total_weight if total_weight > 0 else 0
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for trial: {str(e)}")
                    return float('-inf')
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(n_startup_trials=10)
            )
            
            # Optimize
            n_jobs = 4 if parallel else 1
            study.optimize(objective, n_trials=min(200, self.max_iterations), n_jobs=n_jobs)
            
            # Extract best parameters
            best_params = study.best_params
            
            return {
                'parameters': best_params,
                'best_value': study.best_value,
                'iterations': len(study.trials),
                'converged': study.best_trial is not None
            }
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            raise
    
    def _genetic_optimization(self, strategy_id: str, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform genetic algorithm optimization"""
        try:
            def objective(params):
                # Convert array to parameter dictionary
                parameters = {}
                param_list = list(self.parameter_configs[strategy_id].keys())
                
                for i, param_name in enumerate(param_list):
                    param_config = self.parameter_configs[strategy_id][param_name]
                    
                    if param_config.param_type == 'continuous':
                        parameters[param_name] = params[i]
                    elif param_config.param_type == 'discrete':
                        parameters[param_name] = int(round(params[i]))
                    elif param_config.param_type == 'categorical':
                        idx = int(round(params[i])) % len(param_config.choices)
                        parameters[param_name] = param_config.choices[idx]
                
                try:
                    performance = optimization_data['evaluation_function'](parameters, optimization_data['market_data'])
                    
                    # Calculate weighted objective (negative for minimization)
                    weighted_score = 0
                    total_weight = 0
                    
                    for target_name, target in optimization_data['optimization_targets'].items():
                        if target_name in performance:
                            value = performance[target_name]
                            
                            if target.target_type == 'maximize':
                                score = value
                            elif target.target_type == 'minimize':
                                score = -value
                            else:
                                score = -abs(value - target.target_value)
                            
                            weighted_score += score * target.weight
                            total_weight += target.weight
                    
                    return -(weighted_score / total_weight) if total_weight > 0 else 0  # Negative for minimization
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed: {str(e)}")
                    return float('inf')
            
            # Prepare bounds
            bounds = []
            for param_name, param_config in self.parameter_configs[strategy_id].items():
                if param_config.param_type == 'continuous':
                    bounds.append((param_config.min_value, param_config.max_value))
                elif param_config.param_type == 'discrete':
                    bounds.append((param_config.min_value, param_config.max_value))
                elif param_config.param_type == 'categorical':
                    bounds.append((0, len(param_config.choices) - 1))
            
            # Run differential evolution
            result = differential_evolution(
                objective,
                bounds,
                maxiter=min(100, self.max_iterations // 10),
                popsize=15,
                seed=42
            )
            
            # Convert result back to parameters
            parameters = {}
            param_list = list(self.parameter_configs[strategy_id].keys())
            
            for i, param_name in enumerate(param_list):
                param_config = self.parameter_configs[strategy_id][param_name]
                
                if param_config.param_type == 'continuous':
                    parameters[param_name] = result.x[i]
                elif param_config.param_type == 'discrete':
                    parameters[param_name] = int(round(result.x[i]))
                elif param_config.param_type == 'categorical':
                    idx = int(round(result.x[i])) % len(param_config.choices)
                    parameters[param_name] = param_config.choices[idx]
            
            return {
                'parameters': parameters,
                'best_value': -result.fun,  # Convert back from minimization
                'iterations': result.nit,
                'converged': result.success
            }
            
        except Exception as e:
            logger.error(f"Error in genetic optimization: {str(e)}")
            raise
    
    def _simulated_annealing_optimization(self, strategy_id: str, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform simulated annealing optimization"""
        try:
            def objective(params):
                # Convert array to parameter dictionary (same as genetic algorithm)
                parameters = {}
                param_list = list(self.parameter_configs[strategy_id].keys())
                
                for i, param_name in enumerate(param_list):
                    param_config = self.parameter_configs[strategy_id][param_name]
                    
                    if param_config.param_type == 'continuous':
                        parameters[param_name] = params[i]
                    elif param_config.param_type == 'discrete':
                        parameters[param_name] = int(round(params[i]))
                    elif param_config.param_type == 'categorical':
                        idx = int(round(params[i])) % len(param_config.choices)
                        parameters[param_name] = param_config.choices[idx]
                
                try:
                    performance = optimization_data['evaluation_function'](parameters, optimization_data['market_data'])
                    
                    weighted_score = 0
                    total_weight = 0
                    
                    for target_name, target in optimization_data['optimization_targets'].items():
                        if target_name in performance:
                            value = performance[target_name]
                            
                            if target.target_type == 'maximize':
                                score = value
                            elif target.target_type == 'minimize':
                                score = -value
                            else:
                                score = -abs(value - target.target_value)
                            
                            weighted_score += score * target.weight
                            total_weight += target.weight
                    
                    return -(weighted_score / total_weight) if total_weight > 0 else 0
                    
                except Exception as e:
                    return float('inf')
            
            # Prepare bounds (same as genetic algorithm)
            bounds = []
            for param_name, param_config in self.parameter_configs[strategy_id].items():
                if param_config.param_type == 'continuous':
                    bounds.append((param_config.min_value, param_config.max_value))
                elif param_config.param_type == 'discrete':
                    bounds.append((param_config.min_value, param_config.max_value))
                elif param_config.param_type == 'categorical':
                    bounds.append((0, len(param_config.choices) - 1))
            
            # Run dual annealing
            result = dual_annealing(
                objective,
                bounds,
                maxiter=min(200, self.max_iterations // 5),
                seed=42
            )
            
            # Convert result back to parameters
            parameters = {}
            param_list = list(self.parameter_configs[strategy_id].keys())
            
            for i, param_name in enumerate(param_list):
                param_config = self.parameter_configs[strategy_id][param_name]
                
                if param_config.param_type == 'continuous':
                    parameters[param_name] = result.x[i]
                elif param_config.param_type == 'discrete':
                    parameters[param_name] = int(round(result.x[i]))
                elif param_config.param_type == 'categorical':
                    idx = int(round(result.x[i])) % len(param_config.choices)
                    parameters[param_name] = param_config.choices[idx]
            
            return {
                'parameters': parameters,
                'best_value': -result.fun,
                'iterations': result.nit,
                'converged': result.success
            }
            
        except Exception as e:
            logger.error(f"Error in simulated annealing optimization: {str(e)}")
            raise
    
    def _ensemble_optimization(self, strategy_id: str, optimization_data: Dict[str, Any], parallel: bool = True) -> Dict[str, Any]:
        """Perform ensemble optimization combining multiple methods"""
        try:
            methods = ['bayesian', 'genetic', 'simulated_annealing']
            results = []
            
            if parallel:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {}
                    
                    for method in methods:
                        if method == 'bayesian':
                            future = executor.submit(self._bayesian_optimization, strategy_id, optimization_data, False)
                        elif method == 'genetic':
                            future = executor.submit(self._genetic_optimization, strategy_id, optimization_data)
                        elif method == 'simulated_annealing':
                            future = executor.submit(self._simulated_annealing_optimization, strategy_id, optimization_data)
                        
                        futures[future] = method
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            result['method'] = futures[future]
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Method {futures[future]} failed: {str(e)}")
            else:
                # Sequential execution
                for method in methods:
                    try:
                        if method == 'bayesian':
                            result = self._bayesian_optimization(strategy_id, optimization_data, False)
                        elif method == 'genetic':
                            result = self._genetic_optimization(strategy_id, optimization_data)
                        elif method == 'simulated_annealing':
                            result = self._simulated_annealing_optimization(strategy_id, optimization_data)
                        
                        result['method'] = method
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Method {method} failed: {str(e)}")
            
            if not results:
                raise ValueError("All optimization methods failed")
            
            # Select best result
            best_result = max(results, key=lambda x: x['best_value'])
            
            # Ensemble averaging for robustness
            ensemble_params = {}
            param_values = {param: [] for param in self.parameter_configs[strategy_id].keys()}
            
            for result in results:
                for param_name, value in result['parameters'].items():
                    param_values[param_name].append(value)
            
            for param_name, values in param_values.items():
                param_config = self.parameter_configs[strategy_id][param_name]
                
                if param_config.param_type == 'continuous':
                    ensemble_params[param_name] = np.mean(values)
                elif param_config.param_type == 'discrete':
                    ensemble_params[param_name] = int(round(np.mean(values)))
                elif param_config.param_type == 'categorical':
                    # Use mode for categorical parameters
                    ensemble_params[param_name] = max(set(values), key=values.count)
            
            return {
                'parameters': best_result['parameters'],  # Use best single result
                'ensemble_parameters': ensemble_params,  # Also provide ensemble average
                'best_value': best_result['best_value'],
                'iterations': sum(r['iterations'] for r in results),
                'converged': any(r['converged'] for r in results),
                'method_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble optimization: {str(e)}")
            raise
    
    def _select_optimal_method(self, strategy_id: str, regime: str, optimization_data: Dict[str, Any]) -> str:
        """Select the optimal optimization method based on strategy and market characteristics"""
        try:
            # Strategy complexity
            n_parameters = len(self.parameter_configs[strategy_id])
            
            # Market regime characteristics
            regime_complexity = {
                'low_volatility': 'simple',
                'high_volatility': 'complex',
                'bullish_trend': 'moderate',
                'bearish_trend': 'complex',
                'sideways': 'simple'
            }
            
            complexity = regime_complexity.get(regime, 'moderate')
            
            # Method selection logic
            if n_parameters <= 5 and complexity == 'simple':
                return 'bayesian'
            elif n_parameters > 10 or complexity == 'complex':
                return 'genetic'
            elif complexity == 'moderate':
                return 'simulated_annealing'
            else:
                return 'ensemble'
                
        except Exception as e:
            logger.error(f"Error selecting optimization method: {str(e)}")
            return 'bayesian'
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            
            return macd
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _get_parameter_bounds(self, strategy_id: str) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        bounds = []
        for param_config in self.parameter_configs[strategy_id].values():
            if param_config.param_type == 'continuous':
                bounds.append((param_config.min_value, param_config.max_value))
            elif param_config.param_type == 'discrete':
                bounds.append((param_config.min_value, param_config.max_value))
            elif param_config.param_type == 'categorical':
                bounds.append((0, len(param_config.choices) - 1))
        
        return bounds
    
    def _evaluate_current_performance(self, strategy_id: str, optimization_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate current strategy performance"""
        try:
            current_params = self.strategies[strategy_id]['current_parameters']
            return optimization_data['evaluation_function'](current_params, optimization_data['market_data'])
        except Exception as e:
            logger.error(f"Error evaluating current performance: {str(e)}")
            return {}
    
    def _evaluate_optimized_performance(self, strategy_id: str, parameters: Dict[str, Any], optimization_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate optimized strategy performance"""
        try:
            return optimization_data['evaluation_function'](parameters, optimization_data['market_data'])
        except Exception as e:
            logger.error(f"Error evaluating optimized performance: {str(e)}")
            return {}
    
    def _calculate_improvement_factor(self, current_perf: Dict[str, float], optimized_perf: Dict[str, float]) -> float:
        """Calculate overall improvement factor"""
        try:
            if not current_perf or not optimized_perf:
                return 0.0
            
            # Use primary metric (e.g., Sharpe ratio) for improvement calculation
            primary_metrics = ['sharpe_ratio', 'return', 'profit_factor']
            
            for metric in primary_metrics:
                if metric in current_perf and metric in optimized_perf:
                    current_val = current_perf[metric]
                    optimized_val = optimized_perf[metric]
                    
                    if current_val != 0:
                        return (optimized_val - current_val) / abs(current_val)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating improvement factor: {str(e)}")
            return 0.0
    
    def _calculate_confidence_score(self, optimization_result: Dict[str, Any], optimization_data: Dict[str, Any]) -> float:
        """Calculate confidence score for optimization result"""
        try:
            confidence = 0.5  # Base confidence
            
            # Convergence bonus
            if optimization_result.get('converged', False):
                confidence += 0.2
            
            # Iteration bonus (more iterations = higher confidence)
            max_iterations = self.max_iterations
            actual_iterations = optimization_result.get('iterations', 0)
            iteration_factor = min(actual_iterations / max_iterations, 1.0)
            confidence += 0.2 * iteration_factor
            
            # Ensemble bonus
            if 'method_results' in optimization_result:
                consistency = self._calculate_method_consistency(optimization_result['method_results'])
                confidence += 0.1 * consistency
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _calculate_method_consistency(self, method_results: List[Dict[str, Any]]) -> float:
        """Calculate consistency across different optimization methods"""
        try:
            if len(method_results) < 2:
                return 1.0
            
            # Compare best values
            best_values = [result['best_value'] for result in method_results]
            mean_value = np.mean(best_values)
            std_value = np.std(best_values)
            
            if mean_value == 0:
                return 1.0
            
            cv = std_value / abs(mean_value)  # Coefficient of variation
            consistency = max(0, 1 - cv)  # Lower CV = higher consistency
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating method consistency: {str(e)}")
            return 0.5
    
    def _update_strategy_state(self, strategy_id: str, optimization_result: OptimizationResult) -> None:
        """Update strategy state after optimization"""
        try:
            # Update current parameters
            self.strategies[strategy_id]['current_parameters'] = optimization_result.optimized_parameters.copy()
            self.strategies[strategy_id]['last_optimization'] = datetime.now()
            
            # Store optimization history
            self.optimization_history[strategy_id].append(optimization_result)
            
            # Update parameter importance scores
            self._update_parameter_importance(strategy_id, optimization_result)
            
            logger.info(f"Updated strategy state for {strategy_id}")
            
        except Exception as e:
            logger.error(f"Error updating strategy state: {str(e)}")
    
    def _update_parameter_importance(self, strategy_id: str, optimization_result: OptimizationResult) -> None:
        """Update parameter importance scores based on optimization results"""
        try:
            # Simple importance update based on parameter changes
            current_params = self.strategies[strategy_id]['current_parameters']
            optimized_params = optimization_result.optimized_parameters
            
            for param_name in self.parameter_configs[strategy_id]:
                if param_name in current_params and param_name in optimized_params:
                    param_config = self.parameter_configs[strategy_id][param_name]
                    
                    if param_config.param_type in ['continuous', 'discrete']:
                        current_val = current_params[param_name]
                        optimized_val = optimized_params[param_name]
                        
                        if current_val != 0:
                            change_ratio = abs(optimized_val - current_val) / abs(current_val)
                            # Higher change ratio suggests higher importance
                            param_config.importance_score = 0.9 * param_config.importance_score + 0.1 * change_ratio
                    
        except Exception as e:
            logger.error(f"Error updating parameter importance: {str(e)}")
    
    def _update_learning_models(self, strategy_id: str, optimization_result: OptimizationResult, optimization_data: Dict[str, Any]) -> None:
        """Update machine learning models based on optimization results"""
        try:
            # This is a placeholder for more sophisticated learning
            # In practice, you would update predictive models here
            
            # Store regime-specific optimization results
            regime = optimization_result.market_regime
            if regime not in self.market_regimes:
                self.market_regimes[regime] = []
            
            self.market_regimes[regime].append({
                'strategy_id': strategy_id,
                'parameters': optimization_result.optimized_parameters,
                'performance': optimization_result.performance_metrics,
                'improvement': optimization_result.improvement_factor
            })
            
        except Exception as e:
            logger.error(f"Error updating learning models: {str(e)}")
    
    def should_reoptimize(self, strategy_id: str) -> bool:
        """
        Determine if strategy should be reoptimized
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            True if reoptimization is recommended
        """
        try:
            if strategy_id not in self.strategies:
                return False
            
            last_optimization = self.strategies[strategy_id]['last_optimization']
            
            # Time-based reoptimization
            if last_optimization is None:
                return True
            
            days_since_optimization = (datetime.now() - last_optimization).days
            if days_since_optimization >= self.reoptimization_frequency:
                return True
            
            # Performance-based reoptimization
            # This is a placeholder - in practice you would check actual performance degradation
            performance_history = self.strategies[strategy_id]['performance_history']
            if len(performance_history) >= 5:
                recent_performance = np.mean(performance_history[-5:])
                historical_performance = np.mean(performance_history[:-5]) if len(performance_history) > 5 else recent_performance
                
                if recent_performance < 0.8 * historical_performance:  # 20% degradation threshold
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking reoptimization need: {str(e)}")
            return False
    
    def export_optimization_results(self, strategy_id: str, filename: str) -> None:
        """
        Export optimization results to file
        
        Args:
            strategy_id: Strategy identifier
            filename: Output filename
        """
        try:
            if strategy_id not in self.optimization_history:
                raise ValueError(f"No optimization history for strategy {strategy_id}")
            
            export_data = {
                'strategy_id': strategy_id,
                'optimization_history': [],
                'parameter_configs': {},
                'optimization_targets': {},
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Convert optimization results to serializable format
            for result in self.optimization_history[strategy_id]:
                export_data['optimization_history'].append({
                    'optimization_method': result.optimization_method,
                    'optimized_parameters': result.optimized_parameters,
                    'performance_metrics': result.performance_metrics,
                    'improvement_factor': result.improvement_factor,
                    'confidence_score': result.confidence_score,
                    'market_regime': result.market_regime,
                    'optimization_duration': result.optimization_duration,
                    'iteration_count': result.iteration_count,
                    'convergence_achieved': result.convergence_achieved
                })
            
            # Export parameter configurations
            for param_name, param_config in self.parameter_configs[strategy_id].items():
                export_data['parameter_configs'][param_name] = {
                    'param_type': param_config.param_type,
                    'min_value': param_config.min_value,
                    'max_value': param_config.max_value,
                    'choices': param_config.choices,
                    'current_value': param_config.current_value,
                    'importance_score': param_config.importance_score
                }
            
            # Export optimization targets
            for target_name, target in self.optimization_targets[strategy_id].items():
                export_data['optimization_targets'][target_name] = {
                    'weight': target.weight,
                    'target_type': target.target_type,
                    'target_value': target.target_value,
                    'constraint_min': target.constraint_min,
                    'constraint_max': target.constraint_max
                }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Optimization results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting optimization results: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = AdaptiveOptimizer()
    
    # Example parameter configurations
    param_configs = [
        ParameterConfig('lookback_period', 'discrete', 10, 50, current_value=20),
        ParameterConfig('stop_loss', 'continuous', 0.01, 0.05, current_value=0.02),
        ParameterConfig('take_profit', 'continuous', 0.02, 0.10, current_value=0.04),
        ParameterConfig('strategy_type', 'categorical', choices=['aggressive', 'conservative'], current_value='conservative')
    ]
    
    # Example optimization targets
    optimization_targets = [
        OptimizationTarget('sharpe_ratio', 3.0, 'maximize'),
        OptimizationTarget('max_drawdown', 1.0, 'minimize'),
        OptimizationTarget('total_return', 2.0, 'maximize')
    ]
    
    # Mock evaluation function
    def mock_evaluation_function(parameters, market_data):
        # Simple mock evaluation
        return {
            'sharpe_ratio': np.random.normal(1.5, 0.5),
            'max_drawdown': np.random.uniform(0.05, 0.15),
            'total_return': np.random.normal(0.15, 0.05)
        }
    
    try:
        # Register strategy
        optimizer.register_strategy('test_strategy', param_configs, optimization_targets, mock_evaluation_function)
        
        # Generate sample market data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        market_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0, 1, 252)),
            'volume': np.random.uniform(1000, 5000, 252)
        }, index=dates)
        
        # Perform optimization
        result = optimizer.optimize_strategy('test_strategy', market_data, method='bayesian')
        
        print(f"Optimization completed:")
        print(f"Method: {result.optimization_method}")
        print(f"Improvement: {result.improvement_factor:.2%}")
        print(f"Confidence: {result.confidence_score:.2%}")
        print(f"Optimized parameters: {result.optimized_parameters}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        print(f"Example execution failed: {str(e)}")
